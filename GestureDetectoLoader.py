from ultralytics import YOLO

class GestureDetector:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the gesture detection model
        
        Args:
            model_path (str): Path to the YOLO gesture detection model
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        try:
            self.model = YOLO(model_path)
            print(f"Gesture detection model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading gesture model: {e}")
            self.model = None
    
    def detect(self, frame, conf_threshold=0.3):
        """
        Detect gestures in the given frame
        
        Args:
            frame (numpy.ndarray): Input image
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            list: Detected gestures with their bounding boxes and confidences
        """
        if self.model is None:
            return None
            
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            return results
        except Exception as e:
            print(f"Error during gesture detection: {e}")
            return None