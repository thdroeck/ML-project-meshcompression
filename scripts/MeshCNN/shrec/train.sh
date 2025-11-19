python -m meshcompression.models.train \
    --dataroot assets/shrec_16 \
    --name shrec16 \
    --ncf 64 128 256 256 \
    --pool_res 600 450 300 180 \
    --norm group \
    --resblocks 1 \
    --export_folder meshes \