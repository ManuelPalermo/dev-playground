import subprocess
from pathlib import Path


def test_integration_main(
    project_scripts_folder: Path,
    project_resources_folder: Path,
    device: str,
    tmp_path: Path,
) -> None:
    """Test the main script."""

    # NOTE: for local debugging write to some dir instead of tmp_path
    # tmp_path = Path("/home/palermo/dev-playground/traffic_detection/")

    # GIVEN some config for a traffic detection job
    source_video_path = project_resources_folder / "Video.mp4"
    main_script_path = project_scripts_folder / "main.py"
    outputs_folder = tmp_path / "outputs"
    num_frames = 30
    device = "cpu" if device == "cpu" else "0"
    detector_name = "yolov10x_onnx"

    # WHEN the main script is executed in a subprocess
    result = subprocess.run(
        [
            "python",
            str(main_script_path),
            "--source",
            str(source_video_path),
            "--artefacts_output_dir",
            str(outputs_folder),
            "--device",
            device,
            "--num_frames",
            str(num_frames),
            "--save_frames_to_disk",
            "True",
            "--detector_name",
            detector_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # THEN the script should run without errors
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # THEN the expected output artefacts should have been created
    assert Path(outputs_folder).exists(), "Output artefacts folder was not created"

    # THEN the expected video and num frames should have been written to the output path
    assert Path(outputs_folder / "frames").exists(), "Frames folder was not created"
    assert Path(outputs_folder / "frames" / "output_video.mp4").exists(), "Output video was not created"
    written_frames = list(Path(outputs_folder / "frames").glob("*.jpg"))
    assert len(written_frames) == num_frames, f"Expected {num_frames} frames, but found {len(written_frames)}"

    # THEN the expected history images should have been created
    assert Path(outputs_folder / "history" / "history_pos_2d.png").exists(), "History 2d image was not created"
    assert Path(outputs_folder / "history" / "history_pos_bev.png").exists(), "History bev image was not created"
