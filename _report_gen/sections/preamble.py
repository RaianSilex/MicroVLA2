from build_report import h1, h2, body, bullets


def add(story):
    h1(story, "Shape Legend")
    bullets(story, [
        "<b>B</b> = batch size. <b>N</b> = number of cameras (currently 1). "
        "<b>C</b> = image channels (3 RGB). <b>H, W</b> = image height/width = 240, 320.",
        "<b>k</b> = CHUNK_SIZE = 100 future action timesteps. "
        "<b>state_dim</b> = 8 (ACT) or up to MAX_STATE_DIM = 16 (VLA). "
        "<b>action_dim</b> = 8 (ACT) or up to MAX_ACTION_DIM = 16 (VLA). "
        "<b>D</b> = HIDDEN_DIM = 512. <b>L_dim</b> = LATENT_DIM = 32.",
        "<b>S_img</b> = number of image tokens. ResNet18: ~ 8&times;10 = 80. "
        "DINOv2 ViT-S/14 at 238&times;308: 17&times;22 = 374. "
        "Cellpose pooled: 15&times;20 = 300. Dual encoder total: 374 + 300 = 674.",
        "<b>L_lang</b> = MAX_LANGUAGE_TOKENS = 32. "
        "<b>L_meta</b> = 5 metadata tokens (robot, lab, embodiment, action_type, task_family).",
        "Transformer code uses sequence-first tensors <b>(L, B, D)</b>. "
        "Image batches are <b>(B, N, 3, H, W)</b>. Action chunks are <b>(B, k, action_dim)</b>.",
    ])

    h1(story, "End-to-End Tensor Flow — MicroACT (single robot)")
    bullets(story, [
        "Dataset sample: <b>image (1,3,240,320)</b>, <b>qpos (8,)</b>, "
        "<b>action (100,8)</b>, <b>is_pad (100,)</b>. After DataLoader collation: "
        "<b>image (B,1,3,240,320)</b>, <b>qpos (B,8)</b>, "
        "<b>action (B,100,8)</b>, <b>is_pad (B,100)</b>.",
        "Style encoder path: <b>cls (B,1,512)</b> + <b>qpos_token (B,1,512)</b> + "
        "<b>action_tokens (B,100,512)</b> &rarr; concat to <b>(B,102,512)</b>, "
        "permute to <b>(102,B,512)</b>, encode, take CLS &rarr; project to "
        "<b>mu (B,32)</b>, <b>logvar (B,32)</b>.",
        "Image path: <b>image (B,1,3,240,320)</b> &rarr; flatten cameras to "
        "<b>(B,3,240,320)</b>. Backbone returns either spatial maps "
        "<b>(B,512,Hp,Wp)</b> (single) or token sequence <b>(S,B,512)</b> (dual). "
        "Both routes produce <b>(S_img,B,512)</b>.",
        "Main encoder source: <b>latent_token (1,B,512)</b> + "
        "<b>qpos_token (1,B,512)</b> + <b>img_tokens (S_img,B,512)</b> "
        "&rarr; concat along sequence to <b>(2+S_img,B,512)</b>.",
        "Decoder: query embeddings <b>(100,512)</b> expand to <b>(100,B,512)</b>; "
        "decoder returns <b>(100,B,512)</b>; action head maps to <b>(B,100,8)</b>.",
        "Rollout inference: raw image <b>(H,W,3)</b> + raw state <b>(8,)</b> "
        "&rarr; <b>policy.inference</b> &rarr; action chunk <b>(100,8)</b>. "
        "Temporal aggregation averages overlapping chunks into a single action <b>(8,)</b>; "
        "safety clamp / step limit / EMA preserve shape; two ROS Int32MultiArray messages "
        "of length 5 each (4 axes + speed) are published per stage.",
    ])

    h1(story, "End-to-End Tensor Flow — MicroVLA (multi-robot, language-conditioned)")
    bullets(story, [
        "Dataset sample: <b>image (1,3,240,320)</b>, <b>qpos (16,)</b>, "
        "<b>state_mask (16,)</b>, <b>action (100,16)</b>, <b>action_mask (16,)</b>, "
        "<b>is_pad (100,)</b>, plus scalar long tensors for "
        "<b>robot_id, lab_id, embodiment_id, action_type_id, task_family_id</b> "
        "and a Python <b>instruction</b> string.",
        "Padding: a 4-DOF arm with 4-D action puts its real values in "
        "<b>qpos[:4]</b> and <b>action[:,:4]</b>; the rest is zero. The masks mark which "
        "indices are real so loss and decoder can ignore the zero-padded slots.",
        "Image path: same backbones as ACT; produces <b>(S_img,B,512)</b>.",
        "Language path: instruction strings &rarr; tokenized + frozen DistilBERT &rarr; "
        "<b>(L_lang, B, 512)</b> via a trainable linear projection. "
        "Padding mask <b>(B, L_lang)</b> tells attention which positions are PAD.",
        "Embodiment path: 5 categorical IDs &rarr; 5 learned embeddings stacked to "
        "<b>(5, B, 512)</b>.",
        "Style encoder: cls + padded qpos + padded actions &rarr; "
        "<b>(2+100, B, 512)</b> &rarr; encode &rarr; mu/logvar <b>(B,32)</b>.",
        "Main encoder source: latent <b>(1,B,512)</b> + qpos <b>(1,B,512)</b> + "
        "embodiment <b>(5,B,512)</b> + language <b>(32,B,512)</b> + "
        "image <b>(S_img,B,512)</b> &rarr; concat to <b>(2+5+32+S_img, B, 512)</b>. "
        "Source key padding mask aligns: zeros for fixed tokens, language pad mask for "
        "language tokens, zeros for image tokens.",
        "Decoder + head: <b>(100,512)</b> queries &rarr; <b>(100,B,512)</b> &rarr; "
        "<b>(B,100,16)</b>. Output is multiplied by action_mask so invalid action dims "
        "are zeroed before the loss.",
        "Rollout inference: adapter supplies <b>image_rgb (H,W,3)</b>, "
        "<b>state (state_dim,)</b>, instruction string and metadata IDs. Policy returns "
        "<b>(100, action_dim)</b> in raw absolute units (sliced from the 16-D padded "
        "output). The adapter then runs the same clamp / step-limit / EMA / publish chain.",
    ])
