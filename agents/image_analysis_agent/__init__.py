from .image_classifier import ImageClassifier
# from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        # self.brain_tumor_agent = BrainTumorAgent()
    
    # classify image
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path)
    
    # # brain tumor agent
    # def classify_brain_tumor(self, image_path: str) -> str:
    #     return self.brain_tumor_agent.predict(image_path)
