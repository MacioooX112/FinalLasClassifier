import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import laspy
from typing import Tuple, List, Optional
import os


class PointNetClassifier(nn.Module):
    """
    PointNet-like neural network for classifying point cloud objects from .las files
    """
    
    def __init__(self, num_points: int = 1024, num_classes: int = 10):
        """
        Initialize the classifier
        
        Args:
            num_points: Number of points to sample from each point cloud
            num_classes: Number of classification classes
        """
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        
        # Point feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Transform net for input features
        self.transform_net = TransformNet(K=64)
        
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_points, 3)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Transpose to (batch_size, 3, num_points) for Conv1d
        x = x.transpose(2, 1)
        
        # First MLP: 3 -> 64 -> 64
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Apply transform net
        transform = self.transform_net(x)
        x = x.transpose(2, 1)  # (batch, num_points, 64)
        x = torch.bmm(x, transform)  # Apply transformation
        x = x.transpose(2, 1)  # Back to (batch, 64, num_points)
        
        # Second MLP: 64 -> 64 -> 128 -> 1024
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]  # (batch, 1024, 1)
        x = x.view(batch_size, -1)  # (batch, 1024)
        
        # Classification head
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class TransformNet(nn.Module):
    """
    Transform network to learn affine transformation matrix for rotation/translation invariance
    """
    
    def __init__(self, K: int = 64):
        """
        Initialize transform network
        
        Args:
            K: Dimension of transformation matrix
        """
        super(TransformNet, self).__init__()
        self.K = K
        
        self.conv1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, K * K)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, K, num_points)
            
        Returns:
            Transformation matrix of shape (batch_size, K, K)
        """
        batch_size = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.K, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        x = x.view(batch_size, self.K, self.K) + identity
        
        return x


class PointCloudDataset(Dataset):
    """
    Dataset class for point cloud data
    """
    
    def __init__(self, point_clouds: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            point_clouds: Array of point clouds, shape (n_samples, num_points, 3)
            labels: Array of labels, shape (n_samples,)
        """
        self.point_clouds = point_clouds
        self.labels = labels
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        point_cloud = torch.FloatTensor(self.point_clouds[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return point_cloud, label


class PointCloudClassifierTrainer:
    """
    Trainer class for PointCloudClassifier
    """
    
    def __init__(self, 
                 model: PointNetClassifier,
                 device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: PointNetClassifier model
            device: Device to run training on (cuda or cpu)
        """
        self.model = model
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 50,
              learning_rate: float = 0.001,
              save_path: str = 'best_point_cloud_model.pth'):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save best model
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for point_clouds, labels in train_loader:
                point_clouds = point_clouds.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(point_clouds)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for point_clouds, labels in val_loader:
                        point_clouds = point_clouds.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(point_clouds)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                scheduler.step(avg_val_loss)
                
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), save_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    def predict(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a point cloud
        
        Args:
            point_cloud: Point cloud data, shape (num_points, 3) or (1, num_points, 3)
            
        Returns:
            Class probabilities
        """
        self.model.eval()
        
        # Ensure correct shape
        if point_cloud.ndim == 2:
            point_cloud = np.expand_dims(point_cloud, 0)
        
        point_cloud = torch.FloatTensor(point_cloud).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(point_cloud)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict_class(self, point_cloud: np.ndarray) -> int:
        """
        Predict class label for a point cloud
        
        Args:
            point_cloud: Point cloud data, shape (num_points, 3) or (1, num_points, 3)
            
        Returns:
            Predicted class index
        """
        probabilities = self.predict(point_cloud)
        return np.argmax(probabilities, axis=1)[0]
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for point_clouds, labels in test_loader:
                point_clouds = point_clouds.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(point_clouds)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        # Calculate per-class metrics if sklearn is available
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            
            class_names = ['trees', 'trampoles', 'roads', "rails", "columns", "none"]  # Adjust to your classes
            # Specify labels to include all classes even if not present in predictions
            num_classes = len(class_names)
            report = classification_report(
                all_labels, all_preds, 
                target_names=class_names, 
                labels=list(range(num_classes)),
                zero_division=0
            )
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            
            results = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': all_preds,
                'labels': all_labels
            }
        except ImportError:
            # If sklearn is not available, return basic metrics
            results = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'predictions': all_preds,
                'labels': all_labels
            }
        
        return results
    
    def predict_single_file(self, file_path: str) -> dict:
        """
        Perform a forward pass on a single point cloud file and return the predicted class and probabilities.
        
        Args:
            file_path: Path to a .las file
            
        Returns:
            Dictionary with predicted class and probabilities for the single file
        """
        self.model.eval()

        # Load the point cloud from the .las file
        point_cloud = load_las_file(file_path, num_points=self.model.num_points)
        point_cloud = torch.FloatTensor(point_cloud).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        
        # Perform a forward pass through the model
        with torch.no_grad():
            output = self.model(point_cloud)
            
            # Get predicted class (index of highest probability)
            _, predicted_class = torch.max(output.data, 1)
            
            # Get the class probabilities using softmax
            class_probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()

        # Return the result as a dictionary
        results = {
            'predicted_class': predicted_class.item(),  # Predicted class index
            'predicted_class_name': self.get_class_name(predicted_class.item()),  # Convert index to class name
            'predicted_probabilities': class_probabilities.tolist()  # List of probabilities for each class
        }
        
        return results
    
    def get_class_name(self, class_index: int) -> str:
        """
        Return the name of the class based on the class index.
        
        Args:
            class_index: The index of the predicted class
            
        Returns:
            Class name as a string
        """
        class_names = ['trees', 'trampoles', 'roads', 'rails', 'columns', 'none']
        if 0 <= class_index < len(class_names):
            return class_names[class_index]
        return f'unknown_class_{class_index}'


def augment_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Augment point cloud by rotation, scaling, and adding noise
    Creates variations of the same point cloud for data augmentation
    
    Args:
        points: Point cloud array of shape (n_points, 3)
        
    Returns:
        Augmented point cloud of same shape
    """
    # Make a copy to avoid modifying original
    augmented = points.copy()
    
    # Random rotation around z-axis (vertical axis)
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    augmented = augmented @ rotation_matrix.T
    
    # Random scaling (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)
    augmented = augmented * scale
    
    # Add small random noise (1% of typical point cloud size)
    noise = np.random.normal(0, 0.01, augmented.shape)
    augmented = augmented + noise
    
    return augmented


def load_las_file(file_path: str, num_points: int = 1024) -> np.ndarray:
    """
    Load point cloud from .las file
    
    Args:
        file_path: Path to .las file
        num_points: Number of points to sample (if point cloud has more points)
        
    Returns:
        Point cloud array of shape (num_points, 3) with x, y, z coordinates
    """
    las = laspy.read(file_path)
    
    # Extract x, y, z coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Normalize coordinates
    points = normalize_point_cloud(points)
    
    # Sample or pad to desired number of points
    points = sample_or_pad_points(points, num_points)
    
    return points


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud to unit sphere
    
    Args:
        points: Point cloud array of shape (n_points, 3)
        
    Returns:
        Normalized point cloud
    """
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return points


def farthest_point_sampling(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) to select representative points
    Preserves shape better than random sampling, especially for large point clouds
    Optimized version for large point clouds
    
    Args:
        points: Point cloud array of shape (n_points, 3)
        num_points: Desired number of points
        
    Returns:
        Sampled point cloud of shape (num_points, 3)
    """
    n = len(points)
    if n <= num_points:
        return points
    
    # For very large point clouds, use a faster hybrid approach
    # First randomly downsample to 3x target, then apply FPS
    if n > 10000:
        # Randomly sample to reduce computation
        random_sample_size = min(num_points * 3, n)
        random_indices = np.random.choice(n, random_sample_size, replace=False)
        points = points[random_indices]
        n = len(points)
    
    # Initialize: select first point randomly
    selected_indices = [np.random.randint(0, n)]
    distances = np.full(n, np.inf)
    
    # Iteratively select farthest point from already selected points
    for i in range(num_points - 1):
        last_selected = points[selected_indices[-1]]
        
        # Update distances to nearest selected point (vectorized)
        dists = np.sum((points - last_selected) ** 2, axis=1)
        distances = np.minimum(distances, dists)
        
        # Select point with maximum distance
        selected_indices.append(np.argmax(distances))
        
        # Progress indicator for very large point clouds
        if (i + 1) % 500 == 0 and num_points > 1000:
            print(f"  FPS progress: {i+1}/{num_points-1} points selected", end='\r')
    
    if num_points > 1000:
        print()  # New line after progress
    
    return points[selected_indices]


def sample_or_pad_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Sample or pad points to get exactly num_points
    Uses Farthest Point Sampling for better shape preservation
    
    Args:
        points: Point cloud array of shape (n_points, 3)
        num_points: Desired number of points
        
    Returns:
        Point cloud with exactly num_points
    """
    n = len(points)
    
    if n > num_points:
        # Use Farthest Point Sampling instead of random (better for 6k-20k point clouds)
        return farthest_point_sampling(points, num_points)
    elif n < num_points:
        # Pad by repeating all points cyclically (better than just last point)
        repeat_times = (num_points // n) + 1
        repeated = np.tile(points, (repeat_times, 1))
        return repeated[:num_points]
    else:
        return points


def load_dataset(las_files: List[str], 
                 labels: List[int],
                 num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset of .las files
    
    Args:
        las_files: List of paths to .las files
        labels: List of class labels (integers)
        num_points: Number of points per point cloud
        
    Returns:
        Tuple of (point_clouds, labels) where point_clouds has shape (n_samples, num_points, 3)
    """
    point_clouds = []
    label_array = []
    
    print(f"Loading {len(las_files)} files...")
    for idx, (file_path, label) in enumerate(zip(las_files, labels), 1):
        try:
            print(f"  [{idx}/{len(las_files)}] Loading {file_path}...", end=' ', flush=True)
            pc = load_las_file(file_path, num_points)
            point_clouds.append(pc)
            label_array.append(label)
            print("OK")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print(f"Successfully loaded {len(point_clouds)} point clouds")
    return np.array(point_clouds), np.array(label_array)


def get_class_name(class_index: int) -> str:
    """
    Standalone function to return the name of the class based on the class index.
    
    Args:
        class_index: The index of the predicted class
        
    Returns:
        Class name as a string
    """
    class_names = ['trees', 'trampoles', 'roads', 'rails', 'columns', 'none']
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f'unknown_class_{class_index}'

def modify_classification(las_file, new_classification):
    """
    Modifies the classification of all points in a LAS file.
    
    Args:
        las_file: Path to the LAS file
        new_classification: The new classification value to assign to all points
    """
    try:
        # Open the LAS file
        with laspy.open(las_file) as file:
            las_data = file.read()  # Read the LAS data
            
            # Modify the classification for all points by setting all values to new_classification
            las_data.classification = [new_classification] * len(las_data.points)  # Apply new classification to all points
            
            # Save the modified LAS data to a new file (or overwrite the original file)
            output_path = os.path.join(os.path.dirname(las_file), os.path.basename(las_file))
            las_data.write(output_path)  # Save it to the output path
            
            print(f"Successfully modified the classification and saved to {output_path}")
    except Exception as e:
        print(f"Error modifying {las_file}: {e}")


def classify_folder(dir_path: str, 
                    model_path: str = 'best_ever_model.pth',
                    num_points: int = 2048,
                    num_classes: int = 6,
                    recursive: bool = True) -> dict:
    """
    Load a trained model and classify every .las file in a directory.
    
    Args:
        dir_path: Path to directory containing .las files
        model_path: Path to the trained model file (.pth)
        num_points: Number of points used during training (must match training config)
        num_classes: Number of classes (must match training config)
        recursive: If True, search for .las files recursively in subdirectories
        
    Returns:
        Dictionary with classification results for each file:
        {
            'results': [
                {
                    'file_path': str,
                    'predicted_class': int,
                    'predicted_class_name': str,
                    'confidence': float,
                    'all_probabilities': list
                },
                ...
            ],
            'summary': {
                'total_files': int,
                'successful': int,
                'failed': int,
                'class_distribution': dict
            }
        }
    """
    print(f"Loading model from {model_path}...")
    
    # Initialize model
    model = PointNetClassifier(num_points=num_points, num_classes=num_classes)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return {'results': [], 'summary': {'total_files': 0, 'successful': 0, 'failed': 0, 'error': 'Model file not found'}}
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return {'results': [], 'summary': {'total_files': 0, 'successful': 0, 'failed': 0, 'error': str(e)}}
    
    # Initialize trainer
    trainer = PointCloudClassifierTrainer(model)
    
    # Find all .las files in directory
    print(f"\nSearching for .las files in {dir_path}...")
    las_files = []
    
    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.las'):
                    las_files.append(os.path.join(root, file))
    else:
        if os.path.isdir(dir_path):
            las_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                        if f.lower().endswith('.las')]
        else:
            print(f"ERROR: {dir_path} is not a valid directory")
            return {'results': [], 'summary': {'total_files': 0, 'successful': 0, 'failed': 0, 'error': 'Invalid directory'}}
    
    if len(las_files) == 0:
        print(f"No .las files found in {dir_path}")
        return {'results': [], 'summary': {'total_files': 0, 'successful': 0, 'failed': 0, 'error': 'No .las files found'}}
    
    print(f"Found {len(las_files)} .las file(s)\n")
    
    # Classify each file
    results = []
    successful = 0
    failed = 0
    class_distribution = {}
    
    for idx, las_file in enumerate(las_files, 1):
        print(f"[{idx}/{len(las_files)}] Processing {os.path.basename(las_file)}...", end=' ', flush=True)
        
        try:
            # Predict using trainer
            prediction = trainer.predict_single_file(las_file)
            
            # Extract confidence (max probability)
            confidence = max(prediction['predicted_probabilities'])
            
            # Update class distribution
            class_name = prediction['predicted_class_name']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            
            results.append({
                'file_path': las_file,
                'predicted_class': prediction['predicted_class'],
                'predicted_class_name': class_name,
                'confidence': float(confidence),
                'all_probabilities': prediction['predicted_probabilities']
            })
            modify_classification(las_file, prediction['predicted_class']+19) #asliugdausygdvZXOULYfvw  licyv   ;IYGTBVGTFCTRDHTRWER JYMUGBFGBF
            
            successful += 1
            print(f" {class_name} (confidence: {confidence:.2%})")
            
        except Exception as e:
            failed += 1
            print(f"✗ Error: {e}")
            results.append({
                'file_path': las_file,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total files: {len(las_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nClass Distribution:")
    for class_name, count in sorted(class_distribution.items()):
        print(f"  {class_name}: {count}")
    print("="*60)
    
    return {
        'results': results,
        'summary': {
            'total_files': len(las_files),
            'successful': successful,
            'failed': failed,
            'class_distribution': class_distribution
        }
    }


def evaluate_data(dir_path: str, 
                  model_path: str = 'chwilats.pth',
                  num_points: int = 2048,
                  num_classes: int = 6):
    """
    Alias for classify_folder - evaluates/classifies all .las files in a directory.
    
    Args:
        dir_path: Path to directory containing .las files
        model_path: Path to the trained model file (.pth)
        num_points: Number of points used during training
        num_classes: Number of classes
    """
    return classify_folder(dir_path, model_path, num_points, num_classes, recursive=True)


def real_data():
    """
    Real data damn life so short
    """
    np.random.seed(234234)
    torch.manual_seed(434344)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23476)
    
    num_points = 2048
    num_classes = 6  # 0: trees, 1: trasmpoles, 2: roads , 3: rails, 4:  , 5: none
    
    model = PointNetClassifier(num_points=num_points, num_classes=num_classes)
    # print(model)

    print("Model initialized")

    """ Load dataset """
    las_files = ["Dataset/column1.las", "Dataset/column2.las", 
                "Dataset/metal_pole1.las", "Dataset/rails1.las", "Dataset/rails2.las",
                "Dataset/rails3.las", "Dataset/rails4.las",
                "Dataset/rails5.las", "Dataset/rails6.las",
                "Dataset/rails7.las", "Dataset/rails8.las",
                "Dataset/road1.las", "Dataset/road2.las", "Dataset/road3.las", 
                "Dataset/tram_pole1.las", "Dataset/tram_pole2.las", "Dataset/tram_pole3.las", 
                "Dataset/tram_pole4.las", "Dataset/tram_pole5.las", "Dataset/tram_pole6.las", 
                "Dataset/tree1.las", "Dataset/tree2.las", "Dataset/tree3.las", 
                "Dataset/tree4.las", "Dataset/tt1.las", "Dataset/tt2.las",
                "Dataset/tt3.las", "Dataset/tt4.las", "Dataset/tt5.las"]
    labels = [4,4,5,3,3,3,3,3,3,3,3,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]  # 0: trees, 1: trampoles, 2: roads, 3: rails, 4: columns, 5: none
    """testujemy nowe daty"""
    
    # Shuffle files and labels together
    combined = list(zip(las_files, labels))
    np.random.shuffle(combined)
    las_files, labels = zip(*combined)
    las_files = list(las_files)
    labels = list(labels)

    print("Loading dataset...")
    
    # Load original dataset
    all_data, all_labels = load_dataset(las_files, labels, num_points)
    
    print(f"Original dataset loaded: {len(all_data)} samples")
    
    if len(all_data) == 0:
        print("ERROR: No data loaded! Check your file paths.")
        return
    
    # Augment data: create multiple variations of each point cloud
    augmentation_factor = 5  # Create 5 variations of each original (including original)
    augmented_data = []
    augmented_labels = []
    
    print(f"Creating {augmentation_factor}x augmented dataset...")
    for pc, label in zip(all_data, all_labels):
        # Add original
        augmented_data.append(pc)
        augmented_labels.append(label)
        
        # Add augmented versions
        for _ in range(augmentation_factor - 1):
            augmented_pc = augment_point_cloud(pc.copy())
            augmented_data.append(augmented_pc)
            augmented_labels.append(label)
    
    # Convert to numpy arrays
    all_data = np.array(augmented_data)
    all_labels = np.array(augmented_labels)
    
    print(f"Augmented dataset: {len(all_data)} samples (original: {len(las_files)})")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    # Shuffle the augmented data and labels together
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    print("Data shuffled after augmentation")
    
    # Simple train/test split (80/20)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    train_labels = all_labels[:split_idx]
    test_data = all_data[split_idx:]
    test_labels = all_labels[split_idx:]
    print(test_labels)
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    print("Train and test data split")
    
    train_dataset = PointCloudDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = PointCloudDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    """tutaj ładujemy model"""
    # print("--------------------------------Loading best model--------------------------------")
    # model.load_state_dict(torch.load('BiggestTheLargest.pth'))
    # print("--------------------------------Best model loaded--------------------------------")

    trainer = PointCloudClassifierTrainer(model)


    print("Before training evaluation results")

    print("\n" + "="*50)
    print("BEFORE TRAINING EVALUATION RESULTS")
    print("="*50)
    results = trainer.evaluate(test_loader)

    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    if 'classification_report' in results:
        print("\nClassification Report:")
        print(results['classification_report'])
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
    else:
        print("\n(Install sklearn for detailed classification metrics: pip install scikit-learn)")

    """Tutaj trening i zapisywanie"""

    print("Training model")

    # Train (model auto-saves best model to 'best_point_cloud_model.pth')
    trainer.train(train_loader, epochs=25, learning_rate=0.01, save_path='chwilats.pth')
    
    
    # Evaluate on test set
    print("After training evaluation results")
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    results = trainer.evaluate(test_loader)
    
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    if 'classification_report' in results:
        print("\nClassification Report:")
        print(results['classification_report'])
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
    else:
        print("\n(Install sklearn for detailed classification metrics: pip install scikit-learn)")
    
    # Optionally save final model
    print("Saving model")
    torch.save(model.state_dict(), 'best_ever_model.pth')
    print("\nModel saved to 'best_ever_model.pth'")

def example_usage():
    """
    Example usage of the PointCloudClassifier
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize model
    num_points = 2048  # Increased from 1024 for better preservation of 6k-20k point clouds
    num_classes = 6  # Adjust based on your classes
    
    model = PointNetClassifier(num_points=num_points, num_classes=num_classes)
    
    # Print model summary
    print(model)
    
    # Example: Load dataset
    # las_files = ['path/to/file1.las', 'path/to/file2.las', ...]
    # labels = [0, 1, ...]  # Class labels
    # train_data, train_labels = load_dataset(las_files, labels, num_points)
    
    # Example: Create data loaders
    # train_dataset = PointCloudDataset(train_data, train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Example: Train model
    # trainer = PointCloudClassifierTrainer(model)
    # trainer.train(train_loader, epochs=50, learning_rate=0.001)
    
    # Example: Load saved model and predict
    # model.load_state_dict(torch.load('best_point_cloud_model.pth'))
    # trainer = PointCloudClassifierTrainer(model)
    # point_cloud = load_las_file('path/to/test.las', num_points)
    # prediction = trainer.predict_class(point_cloud)
    # print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    evaluate_data("Clusters/")
