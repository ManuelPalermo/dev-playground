import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from traffic_detection.configs.config_factory import KNOW_SITE_CONFIGS, get_site_config
from traffic_detection.definitions import Box2D
from traffic_detection.detection.detector_factory import VALID_DETECTOR_MODELS, get_detector_model
from traffic_detection.object_properties.object_properties_classification import (
    ObjectPropertiesClassification,
)
from traffic_detection.object_properties.vehicle_selector import (
    VehiclesOfInterestSelector,
)
from traffic_detection.perspective_transformation.cam_to_bev import (
    CameraToBevTransformation,
)
from traffic_detection.tracking.history_and_velocity_manager import TrackHistoryManager
from traffic_detection.tracking.sort import Sort
from traffic_detection.utils.box2d import compute_boxes_centers_bottom_from_boxes_xyxy
from traffic_detection.utils.image import draw_boxes, draw_keypoints, draw_polygons, resize_img


def process_video(opt: argparse.Namespace) -> None:  # noqa: PLR0915
    """Main function to process video and detect vehicles of interest."""
    t0_all = time.time()
    videoSrcPath = opt.source
    if not Path(videoSrcPath).exists():
        print(f" Exit as the video path {videoSrcPath} doesnt exist")
        return
    cap = cv2.VideoCapture(videoSrcPath)
    frames_count, fps, width, height = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT),
        cap.get(cv2.CAP_PROP_FPS),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    width = int(width)
    height = int(height)

    print(f"Input video #frames ={frames_count}, fps ={fps}, width ={width}, height={height}")

    outputs_folder = Path(opt.artefacts_output_dir)
    print("Saving output artifacts to: ", outputs_folder)

    site_config = get_site_config(opt.site_config)

    detector2d = get_detector_model(
        model_name=opt.detector_name,
        device="cpu" if opt.device == "cpu" else f"cuda:{opt.device}",
        apply_nms=True,
    )

    cam2bev_transformer = CameraToBevTransformation(
        source=site_config["perspective_area_pixels"],
        target=site_config["perspective_area_world"],
    )

    # TODO: use extended kalman filter to track in pixel space (non-linear due to camera view/distance)
    #      or project raw boxes to BEV and track in BEV space (closer to linear)
    tracker = Sort(
        iou_threshold=0.5,
        min_age=3,
        max_age=5,
        min_age_predict=int(fps / 2),  # predict only boxes with >0.5 seconds of tracking (for vel to stabilize)
        dt=1.0 / fps,
    )
    track_history_manager = TrackHistoryManager(max_length=30, dt=1.0 / fps, velocity_filter_alpha=0.1)
    # track_prediction_manager = TrackHistoryManager(max_length=30, dt=1.0 / fps, velocity_filter_alpha=0.1)

    properties_classifier = ObjectPropertiesClassification()

    polygons_mapping = site_config["count_area_polygons"]
    vehicle_selector = VehiclesOfInterestSelector(
        areas_of_interest_polygon=polygons_mapping,
        valid_labels=None,  # ["car"],
        valid_colors=None,  # ["white"],
        # dont account for tracks with low confidence or low tracking history
        min_score=0.05,
        min_track_age=5,
    )
    frameNumber = 0
    max_num_frames = min(opt.num_frames, int(frames_count))
    while cap.isOpened() and frameNumber < max_num_frames:
        ret, frame = cap.read()

        if ret:
            # opencv reads images in BGR format by default, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_bev = cam2bev_transformer.warp_image(frame)  # warp image to BEV for visualization

            # ----------------------- predict on image ------------------------
            # detect 2d boxes
            frame_for_det = resize_img(frame, max_size=int(opt.img_size))
            box2d_raw = detector2d(frame_for_det)
            img_resize_fx = frame.shape[1] / frame_for_det.shape[1]
            img_resize_fy = frame.shape[0] / frame_for_det.shape[0]
            resized_boxes = box2d_raw.boxes * [img_resize_fx, img_resize_fy, img_resize_fx, img_resize_fy]

            # compute the bev positions of the boxes inside
            boxes_center_bev_pos = cam2bev_transformer.transform_points(
                points_uv=compute_boxes_centers_bottom_from_boxes_xyxy(resized_boxes)
            )
            box2d = Box2D(
                boxes=resized_boxes,
                labels=box2d_raw.labels,
                scores=box2d_raw.scores,
                bev_pos=boxes_center_bev_pos,
            )

            # track boxes
            box2d_tracked = tracker.update(box2d)

            # infer object properties
            box2d_tracked_properties = properties_classifier(detections=box2d_tracked, image=frame)

            # update track history
            box2d_tracked_properties_history = track_history_manager.update(box2d_tracked_properties)

            # select boxes in areas of interest with desired properties
            box2d_tracked_properties_history_selected = vehicle_selector(box2d_tracked_properties_history)
            box2d_selected_area_counts = vehicle_selector.get_areas_of_interest_counts()

            # predict boxes next states (based on tracker model)
            box2d_predicted = tracker.predict_future_state(future_time=2.0)  # predict s into the future of boxes
            # box2d_predicted_properties_history = track_prediction_manager.update(box2d_predicted)
            box2d_predicted_properties_history_selected = vehicle_selector(box2d_predicted)

            # -------------------------visualize results ----------------------

            if opt.save_frames_to_disk or opt.visualize_frames:
                # visualize image with only selected boxes inside area of interest
                image_vis = cv2.cvtColor(np.copy(frame), cv2.COLOR_RGB2BGR)  # convert back to BGR for opencv vis
                image_vis = draw_polygons(
                    image_vis, polygons=[site_config["perspective_area_pixels"]], color=(0, 0, 255)
                )
                cv2.putText(  # draw frame counter
                    image_vis,
                    "Frame#: " + str(frameNumber),
                    (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (2, 10, 200),
                    2,
                )
                for idx, area_name in enumerate(polygons_mapping.keys()):
                    image_vis = draw_boxes(
                        image_vis,
                        box2d_tracked_properties_history_selected[area_name],
                        box_color=(0, 255, 0),
                        fastest_box_color=(255, 0, 255),
                    )
                    image_vis = draw_boxes(
                        image_vis,
                        box2d_predicted_properties_history_selected[area_name],
                        box_color=(127, 0, 0),
                    )

                    image_vis = draw_polygons(image_vis, polygons=[polygons_mapping[area_name]], color=(255, 0, 0))
                    cv2.putText(  # draw number of boxes in area of interest
                        image_vis,
                        f"{area_name}: {len(box2d_selected_area_counts[area_name])}",
                        (0, 70 + idx * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 200, 0),
                        2,
                    )

                # visualize BEV image
                image_bev_vis = cv2.cvtColor(np.copy(frame_bev), cv2.COLOR_RGB2BGR)
                image_bev_vis = cv2.flip(image_bev_vis, 0)
                scaled_boxes_center_bev_pos = boxes_center_bev_pos / cam2bev_transformer.resolution_m_per_px
                image_bev_vis = draw_keypoints(image_bev_vis, scaled_boxes_center_bev_pos, color=(0, 255, 0))
                image_bev_vis = cv2.flip(image_bev_vis, 0)

                # join both 2d image and bev image together for easier viualization
                bev_height, bev_width = image_bev_vis.shape[:2]
                vis_height, _ = image_vis.shape[:2]
                scale_factor = vis_height / bev_height
                new_bev_width = int(bev_width * scale_factor)
                image_bev_resized = cv2.resize(image_bev_vis, (new_bev_width, vis_height))
                combined_image_vis = np.hstack((image_vis, image_bev_resized))

            # save frame to disc (disable for realtime inference)
            if opt.save_frames_to_disk:
                frame_save_path = outputs_folder / "frames" / f"frame_{frameNumber:06d}.jpg"
                frame_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_save_path), combined_image_vis)

            if opt.visualize_frames:
                # visualize in window
                cv2.imshow("output_result", combined_image_vis)
                # cv2.imshow("output_image", image_vis)
                # cv2.imshow("output_bev", image_bev_vis)
                key = cv2.waitKey(1)
                # Quit when 'q' is pressed
                if key == ord("q"):
                    break
                if key == ord("k"):
                    cv2.waitKey(0)
        else:
            print(f"coudn't read current frame #{frameNumber}")
        frameNumber = frameNumber + 1

    cap.release()
    cv2.destroyAllWindows()
    t1_all = time.time()
    print(f"Done. process_video took ({t1_all - t0_all:.3f}s)")

    # save tracks history
    track_history_manager.plot_history(savedir=outputs_folder / "history/")
    # track_prediction_manager.plot_history(savedir=outputs_folder / "history_preds/")

    # create video from all the saved .jpg frames saved in the frames/ folder
    if opt.save_frames_to_disk:
        video_save_path = outputs_folder / "frames" / "output_video.mp4"
        print(f"Creating video from saved frames to {video_save_path}")
        os.system(
            f"ffmpeg -y -framerate {fps} -i {outputs_folder / 'frames' / 'frame_%06d.jpg'} -c:v libx264 -pix_fmt yuv420p {video_save_path}"
        )
        print(f"Video saved to {video_save_path}")


def get_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="program to open a video and display; ",
        formatter_class=argparse.RawTextHelpFormatter,
        usage='\n #1: open a single video: >> python3 main.py -s "videoname.MP4"',
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=False,
        help="source",
        default="./data/Video.mp4",
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--device",
        default="0",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=300,  # 999999999,
        help="maximum number of frames to process",
    )
    parser.add_argument(
        "--artefacts_output_dir",
        type=str,
        default="./outputs/",
        help="if predictions of frames should be saved to disc",
    )
    parser.add_argument(
        "--site_config",
        type=str,
        default="demo",
        choices=KNOW_SITE_CONFIGS,
        help="site configuration to use with areas of inteferest layouts",
    )
    parser.add_argument(
        "--visualize_frames",
        type=bool,
        default=False,
        help="if predictions of frames should be visualized in a window",
    )
    parser.add_argument(
        "--save_frames_to_disk",
        type=bool,
        default=False,
        help="if predictions of frames should be saved to disc",
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        default="yolov10x_onnx",  # onnx based, fast and reasonable detections
        # default="owlv2_base_patch16_ensemble", # very good detections and open label set, but slow
        # default="PekingU/rtdetr_v2_r50vd",  # good detections and reasonably fast
        help="name of the detector to use",
        choices=VALID_DETECTOR_MODELS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    print("\nArguments passed:")
    for arg, value in vars(args).items():
        print(f" - {arg}: {value}")
    print("-------------------")

    process_video(args)
    print("## Exit out of the program .......")
