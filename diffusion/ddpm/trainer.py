class Trainer:
    def __init__():
        pass

    def train_one_epoch(
        model,
        loader,
        sd,
        optimizer,
        scaler,
        loss_fn,
        epoch=800,
        base_config=BaseConfig(),
        training_config=TrainingConfig(),
    ):

        loss_record = MeanMetric()
        model.train()

        with tqdm(total=len(loader), dynamic_ncols=True) as tq:
            tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

            for x0s, _ in loader:  # line 1, 2
                tq.update(1)

                ts = torch.randint(
                    low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE
                )  # line 3
                xts, gt_noise = sd.forward_diffusion(x0s, ts)  # line 4

                with amp.autocast():
                    pred_noise = model(xts, ts)
                    loss = loss_fn(gt_noise, pred_noise)  # line 5

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                loss_value = loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

            mean_loss = loss_record.compute().item()

            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

        return mean_loss
