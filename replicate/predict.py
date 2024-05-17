from cog import BasePredictor, Path, Input

from PIL import Image
from hair_segment_predictor import HairSegmentPredictor

class Predictor(BasePredictor):
    def setup(self):
        self.hair_segment_predictor = HairSegmentPredictor()
        self.hair_segment_predictor.setup()

    def predict(
            self,
            image: Path = Input(description="Image of a dragon, or notdragon:")
    ) -> Path:
        img, b_mask = self.hair_segment_predictor.find_mask(str(image))
        for x in range(len(b_mask)):
            for y in range(len(b_mask[0])):
                if b_mask[x][y] == 255:
                    img[x][y] = [255,255,255]

        Image.fromarray(img).save('/tmp/output.jpg')

        return Path('/tmp/output.jpg')

