import time
import torch

from queue import Queue
from threading import Thread

from ultralytics import YOLO

class YOLOv10Detector:
    def __init__(self, input_size, device='cuda'):
        """
        Initialize YOLOv10 model for human detection
        
        Args:
            input_size (int): Input size for the model
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.input_size = input_size
        self.device = device
        # Load YOLOv10-n model
        self.model = YOLO('yolov10n.pt')
        self.classes = self.model.names
        
        # Focus on the 'person' class (typically class 0 in COCO dataset)
        self.person_class_id = 0
        
        print(f"YOLOv10-n model loaded on {device}")
        
    def detect(self, image, need_resize=True, expand_bb=10):
        """
        Detect humans in the image
        
        Args:
            image (numpy.ndarray): Input image
            need_resize (bool): Whether to resize the image before detection
            expand_bb (int): Pixels to expand the bounding box by
            
        Returns:
            torch.Tensor: Detection results with format [x1, y1, x2, y2, confidence, class_id]
        """
        # Run the model with confidence threshold to reduce false positives
        results = self.model(image, verbose=False, conf=0.80)
        
        # Process results
        detections = []
        
        for result in results:
            # Get the boxes from the results
            boxes = result.boxes
            
            # Filter for only person class
            person_indices = (boxes.cls == self.person_class_id).nonzero().squeeze(-1)
            
            if len(person_indices) > 0:
                # Extract person detections
                person_boxes = boxes[person_indices]
                
                # Format: [x1, y1, x2, y2, confidence, class_id]
                for i in range(len(person_boxes)):
                    box = person_boxes.xyxy[i].tolist()  # Get box coordinates (x1, y1, x2, y2)
                    conf = person_boxes.conf[i].item()   # Get confidence
                    
                    # Expand bounding box if requested
                    if expand_bb > 0:
                        box[0] = max(0, box[0] - expand_bb)
                        box[1] = max(0, box[1] - expand_bb)
                        box[2] = min(image.shape[1], box[2] + expand_bb)
                        box[3] = min(image.shape[0], box[3] + expand_bb)
                    
                    # Add detection [x1, y1, x2, y2, confidence, class_id]
                    # Only 6 elements to match the expected format in the tracker
                    detections.append(box + [conf])
        
        # Convert to tensor format like the original implementation
        if len(detections) > 0:
            return torch.tensor(detections, dtype=torch.float32)
        else:
            return None


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()