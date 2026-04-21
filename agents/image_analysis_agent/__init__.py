from .image_classifier import ImageClassifier
from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent
from .brain_stroke_agent.brain_stroke_inference import BrainStrokeAgent


class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """

    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)

        # Reserved-interface CV agents (actual algorithms implemented by another team)
        self.brain_tumor_agent = BrainTumorAgent(
            model_path=getattr(config.medical_cv, "brain_tumor_model_path", None),
            output_dir=getattr(config.medical_cv, "brain_tumor_output_dir", None),
        )
        self.brain_stroke_agent = BrainStrokeAgent(
            model_path=getattr(config.medical_cv, "brain_stroke_model_path", None),
            output_dir=getattr(config.medical_cv, "brain_stroke_output_dir", None),
        )
        self.brain_tumor_output_path = getattr(
            config.medical_cv,
            "brain_tumor_output_path",
            "./uploads/brain_tumor_output/brain_tumor_plot.png",
        )
        self.brain_stroke_output_path = getattr(
            config.medical_cv,
            "brain_stroke_output_path",
            "./uploads/brain_stroke_output/brain_stroke_plot.png",
        )

    # classify image
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path)

    # brain tumor agent (reserved interface)
    def detect_brain_tumor(self, image_path: str) -> dict:
        return self.brain_tumor_agent.predict(image_path, self.brain_tumor_output_path)

    # brain stroke agent (reserved interface)
    def detect_brain_stroke(self, image_path: str) -> dict:
        return self.brain_stroke_agent.predict(image_path, self.brain_stroke_output_path)
