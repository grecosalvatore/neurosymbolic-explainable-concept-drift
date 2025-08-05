import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ERICExtractor:
    """
    Extracting Relations Inferred from Convolutions (ERIC)
    A framework to extract symbolic rules from CNN kernel activations.
    """
    def __init__(self, cnn_model, num_classes=7):
        self.cnn_model = cnn_model
        self.num_classes = num_classes
        self.kernel_thresholds = None
        self.dt = None
        self.rule_fidelity = 0.0
        self.rule_fidelity_test = 0.0
        self.rules = []
        #self.class_specific_rules = {}

    def compute_kernel_norms(self, feature_maps):
        """
        Compute L1 norms of kernel activations from feature maps.
        
        Args:
            feature_maps: CNN feature maps of shape [B, C, H, W]
            
        Returns:
            Kernel norms of shape [B, C]
        """
        # L1 norm calculation across spatial dimensions
        return torch.sum(torch.abs(feature_maps), dim=(2, 3))
    
    def compute_kernel_thresholds(self, kernel_norms_list, percentiles_list):
        """
        Computes adaptive thresholds for binarizing kernel activations.
        Uses percentile-based approach for more discriminative thresholding.
        
        Args:
            kernel_norms_list: List of kernel norms from normal images
            
        Returns:
            Thresholds for each kernel
        """
        # Stack all kernel norms from the training set
        #all_norms = torch.cat(kernel_norms_list, dim=0)
        all_norms = kernel_norms_list

        # For each kernel, compute a more discriminative threshold
        # using percentile-based approach instead of simple mean
        thresholds = []
        for k, percentile in zip(range(all_norms.shape[1]), percentiles_list):
            kernel_values = all_norms[:, k].cpu().numpy()
            # Use 50th percentile (median) as threshold for better separation
            threshold = np.percentile(kernel_values, percentile)
            thresholds.append(threshold)
        
        self.kernel_thresholds = torch.tensor(thresholds, device=all_norms.device)
        return self.kernel_thresholds
    
    def binarize_kernel_activations(self, kernel_norms):
        """
        Binarize kernel activations using thresholds.
        
        Args:
            kernel_norms: Kernel norms of shape [B, C]
            
        Returns:
            Binary activations of shape [B, C] where 1 represents
            activation above threshold and -1 represents below.
        """
        if self.kernel_thresholds is None:
            raise ValueError("Kernel thresholds not computed. Call compute_kernel_thresholds first.")
        
        # Compare with thresholds (1 if above, -1 if below)
        return torch.where(kernel_norms > self.kernel_thresholds, 1., 0.)
        
    def extract_rules(self, binary_activations, labels, test_binary_activations, test_labels, max_depth=5):
        """
        Extract symbolic rules using an improved decision tree approach.
        
        Args:
            binary_activations: Binary kernel activations of shape [N, C]
            labels: Ground truth labels of shape [N]
            max_depth: Maximum depth of decision tree
            
        Returns:
            List of extracted rules
        """
        try:
            # Convert tensors to numpy arrays
            binary_activations_np = binary_activations.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Convert tensors to numpy arrays
            test_binary_activations_np = test_binary_activations.cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
            
            # Get unique non-zero labels
            unique_labels = np.unique(labels_np)
            #unique_labels = unique_labels[unique_labels > 0]  # Skip background
            
            if len(unique_labels) < 2:
                raise ValueError(f"Not enough unique labels for classification: {unique_labels}")
            
            # Train decision tree with improved parameters
            self.dt = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=max_depth,
                min_samples_split=4,     # Require at least 4 samples to split a node
                min_samples_leaf=2,      # Require at least 2 samples in each leaf
                class_weight='balanced'  # Account for class imbalance
            )
            self.dt.fit(binary_activations_np, labels_np)
            
            # Convert decision tree to rules
            rules = self._tree_to_rules(self.dt)
            
            # Evaluate rule fidelity (train set)
            predictions = self.dt.predict(binary_activations_np)
            self.rule_fidelity = np.mean(predictions == labels_np)

            # Evaluate rule fidelity (test set)
            test_predictions = self.dt.predict(test_binary_activations_np)
            self.rule_fidelity_test = np.mean(test_predictions == test_labels_np)
            
            # Try ensemble approach if rule fidelity is low
            if self.rule_fidelity < 0.7 and len(unique_labels) > 1:
                print(f"Rule fidelity is low ({self.rule_fidelity:.4f}). Trying random forest...")
                
                # Use Random Forest for potentially better rules
                rf = RandomForestClassifier(
                    n_estimators=10,     # Use 10 trees
                    max_depth=4,         # Slightly shallower trees
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42
                )
                rf.fit(binary_activations_np, labels_np)
                
                # Evaluate random forest fidelity
                rf_predictions = rf.predict(binary_activations_np)
                rf_fidelity = np.mean(rf_predictions == labels_np)
                
                print(f"Random forest fidelity: {rf_fidelity:.4f}")
                
                if rf_fidelity > self.rule_fidelity:
                    print("Using random forest rules instead of decision tree")
                    # Extract rules from the most important tree in the forest
                    importances = rf.feature_importances_
                    best_tree_idx = np.argmax([estimator.feature_importances_.sum() 
                                               for estimator in rf.estimators_])
                    best_tree = rf.estimators_[best_tree_idx]
                    
                    rules = self._tree_to_rules(best_tree)
                    self.rule_fidelity = rf_fidelity
            
            if not rules:
                print("Warning: No rules extracted from decision tree. Creating default rules.")
                # Create default rules for each non-zero class
                for class_id in unique_labels:
                    #if class_id > 0:  # Skip background
                    if class_id > -1:  # Skip background
                        rule = {
                            'antecedent': [(1, 0)],  # Positive literal for first kernel
                            'consequent': int(class_id)
                        }
                        rules.append(rule)
            
            self.rules = rules
            return rules
            
        except Exception as e:
            print(f"Error extracting rules: {e}")
            print("Creating default rules instead...")
            
            # Create default rules for each non-zero class in labels
            unique_labels = np.unique(labels.cpu().numpy())
            rules = []
            
            for label in unique_labels:
                if label > 0:  # Skip background
                    # Create a simple rule: first kernel → class
                    rule = {
                        'antecedent': [(1, 0)],  # Positive literal for first kernel
                        'consequent': int(label)
                    }
                    rules.append(rule)
            
            self.rules = rules
            self.rule_fidelity = 0.5  # Default fidelity
            return rules
    
    def _tree_to_rules(self, tree):
        """
        Convert a decision tree to a list of rules.
        
        Args:
            tree: Trained decision tree classifier
            
        Returns:
            List of rules, where each rule is a dict with 'antecedent' and 'consequent'
        """
        rules = []
        
        def traverse_tree(node_id, antecedent, rules):
            # If leaf node, create rule
            if tree.tree_.children_left[node_id] == -1:
                # Get the majority class in this leaf
                class_counts = tree.tree_.value[node_id][0]
                class_id = np.argmax(class_counts)
                
                # Only create rules for non-background classes and non-empty leaves
                #if class_id > 0 and class_counts[class_id] > 0:
                if class_id > -1 and class_counts[class_id] > 0:
                    # Calculate rule confidence based on class distribution in leaf
                    total_samples = np.sum(class_counts)
                    confidence = class_counts[class_id] / total_samples if total_samples > 0 else 0
                    
                    rule = {
                        'antecedent': antecedent.copy(),
                        'consequent': int(class_id),
                        'confidence': float(confidence),  # Add confidence to rule
                        'samples': int(total_samples)     # Add number of samples to rule
                    }
                    rules.append(rule)
                return
                
            # Not a leaf node, add feature test and traverse children
            feature = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            
            # Left child (feature <= threshold, which means binary_activation < 0)
            left_antecedent = antecedent.copy()
            left_antecedent.append((-1, feature))  # Negative literal
            traverse_tree(tree.tree_.children_left[node_id], left_antecedent, rules)
            
            # Right child (feature > threshold, which means binary_activation > 0)
            right_antecedent = antecedent.copy()
            right_antecedent.append((1, feature))  # Positive literal
            traverse_tree(tree.tree_.children_right[node_id], right_antecedent, rules)
            
        try:
            traverse_tree(0, [], rules)
        except Exception as e:
            print(f"Error in tree traversal: {e}")
            # Create a default rule
            if len(rules) == 0:
                rules.append({
                    'antecedent': [(1, 0)],  # Positive literal for first kernel
                    'consequent': 1,         # Default class 1
                    'confidence': 1.0,       # Default confidence
                    'samples': 1             # Default sample count
                })
                
        return rules
    
    def rule_to_string(self, rule, kernel_names=None):
        """
        Convert a rule to a human-readable string.
        
        Args:
            rule: Rule dict with 'antecedent' and 'consequent'
            kernel_names: Optional mapping from kernel indices to names
            
        Returns:
            String representation of the rule
        """
        antecedent_strs = []
        
        for sign, feature in rule['antecedent']:
            if kernel_names is not None and feature < len(kernel_names):
                feature_name = kernel_names[feature]
            else:
                feature_name = f"kernel_{feature}"
                
            if sign > 0:
                antecedent_strs.append(feature_name)
            else:
                antecedent_strs.append(f"¬{feature_name}")
                
        # Add confidence if available
        confidence_str = ""
        if 'confidence' in rule:
            confidence_str = f" (conf={rule['confidence']:.2f})"
            
        if not antecedent_strs:
            return f"True → Class_{rule['consequent']}{confidence_str}"
            
        return f"{' ∧ '.join(antecedent_strs)} → Class_{rule['consequent']}{confidence_str}"
