import json
from pathlib import Path
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
rows = []
for target in ((1280, 1280), (1280, 704), (640, 1920)):
    for center in (True, False):
        source = np.zeros((333, 1001, 3), dtype=np.uint8)
        image = LetterBox(new_shape=target, auto=True, stride=32, center=center)(image=source)
        gain = min(target[0] / source.shape[0], target[1] / source.shape[1])
        resized = (round(source.shape[1] * gain), round(source.shape[0] * gain))
        dw = (target[1] - resized[0]) % 32
        dh = (target[0] - resized[1]) % 32
        left = round(dw / 2 - 0.1) if center else 0
        top = round(dh / 2 - 0.1) if center else 0
        boxes = torch.tensor([[0.0, 0.0, 1001.0, 333.0], [101.0, 51.0, 501.0, 251.0]])
        network = boxes * gain + torch.tensor([left, top, left, top])
        restored = scale_boxes(image.shape[:2], network.clone(), source.shape[:2], ratio_pad=((gain, gain), (left, top)))
        error = float((restored - boxes).abs().max())
        assert error < 0.001, error
        rows.append(dict(target=target, center=center, input_shape=image.shape[:2], gain=gain, padding_ltrb=[left, top, dw - left, dh - top], max_abs=error))
Path('evidence-shapes/letterbox.json').write_text(json.dumps(rows, indent=2) + '\n')
print('LETTERBOX_RESTORE_PASS', len(rows))
