import CheckpointHelper.bucket_handler as bh
import io
bucket = "training_model_checkpoints"

def save_checkpoint(models, optimizer, epoch, loss, model_name, traininig_id):

    byte_buffer = io.BytesIO()
    
    model_state_dicts = {}

    for key in models:
        model_state_dicts[key] = models[key].state_dict()

    torch.save({
        "model_state_dicts": model_state_dicts,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, byte_buffer)

    bh.upload_to_bucket(bucket, "/"+model_name+"/"+traininig_id+"/checkpoint.bin", byte_buffer)


def load_checkpoint(models, optimizer, model_name, training_id):

    byte_buffer = bh.get_bytes_from_blob(bucket, "/"+model_name+"/"+traininig_id+"/checkpoint.bin")

    data_dict = torch.load(byte_buffer)

    model_state_dicts = data_dict["model_state_dicts"]

    for key in model_state_dicts:
        models[key].load_state_dict(model_state_dicts[key])

    optimizer.load_state_dict(data_dict["optimizer"])
