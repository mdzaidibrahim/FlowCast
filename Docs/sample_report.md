VISVESVARAYA TECHNOLOGICAL UNIVERSITY
Jnana Sangama, Belagavi-590018
Deep Learning Project Report on
“CROP CARE: DEEP LEARNING BASED PLANT DISEASE
DETECTION SYSTEM”
Submitted in Partial fulfilment of the Requirements for the Degree of
Bachelor of Engineering in Computer Science and Engineering (Artificial Intelligence
& Machine Learning)
By
AKSHITA N NAIR (1CR22CI003)
HARSHINI R (1CR22CI020)
Under the Guidance of,
Ms Navya VK
Assistant Professor, Dept. of AIML
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (AI & ML)
CMR INSTITUTE OF TECHNOLOGY
Affiliated to VTU, Approved by AICTE, Accredited by NBA and NAAC with “A++” Grade
ITPL MAIN ROAD, BROOKFIELD, BENGALURU-560037, KARNATAKA, INDIA
1
CMR INSTITUTE OF TECHNOLOGY
Affiliated to VTU, Approved by AICTE, Accredited by NBA and NAAC with “A++” Grade
ITPL MAIN ROAD, BROOKFIELD, BENGALURU-560037, KARNATAKA, INDIA
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (AI & ML)
CERTIFICATE
This is to certify that the Assignment report work entitled “Crop Care: Deep Learning Based
Plant Disease Detection System” has been carried out by Akshita N Nair (1CR22CI003) and
Harshini R (1CR22CI020), Bonafide students of CMR Institute of Technology, Bengaluru in
partial fulfilment for the award of the Degree of Bachelor of Engineering in Computer
Science and Engineering (Artificial Intelligence and Machine Learning) of the
Visvesvaraya Technological University, Belagavi during the year 2025-2026. It is certified that
all corrections/suggestions indicated for the Internal Assessment have been incorporated in the
report deposited in the departmental library. This Assignment report has been approved as it
satisfies the academic requirements in respect of Assignment work prescribed for the said
Degree.
Signature of
Prof. Navya VK
Project Guide
Dept. of AI&ML
CMR Institute of Technology
Signature of
Dr. Shyam P Joy
Head of Department
Dept. of AI&ML
CMR Institute of Technology
2
DECLARATION
Ms. Akshita N Nair (1CR22CI003) and Ms. Harshini R (1CR22CI020) hereby declare that the
Assignment report entitled “Crop Care: Deep Learning Based Plant Disease Detection System”
has been carried out by me under the guidance of Prof. Navya VK, Assistant Professor, Department
of AI&ML, CMR Institute of Technology, Bengaluru, in partial fulfilment of the requirement for the
degree of BACHELOR OF COMPUTER SCIENCE AND ENGINEERING (ARTIFICIAL
INTELLIGENCE AND MACHINE LEARNING), of Visvesvaraya Technological University,
Belgaum during the academic year 2025-2026. The work done in this Assignment report is original
and it has not been submitted for any other degree in any university.
Place: Bangalore
Date: 01/11/2025
Akshita N Nair (1CR22CI003)
Harshini R (1CR22CI020)
3
ABSTRACT
Early detection of plant diseases plays a crucial role in improving agricultural productivity and
minimizing economic losses. Traditional diagnostic methods rely heavily on expert observation and
laboratory analysis, which are often time-consuming, costly, and inaccessible to small-scale farmers.
In this context, the proposed Crop Care system leverages the power of deep learning to automatically
identify plant diseases from leaf images, enabling rapid and reliable diagnosis through intelligent
image classification.
The core of this project is a Convolutional Neural Network (CNN) model trained to classify crop leaf
images into multiple disease categories across several plant types. The model architecture was
designed to progressively extract low- to high-level visual features using stacked convolution and
pooling layers, followed by dense layers for classification.
Techniques such as data augmentation, dropout regularization, and Adam optimization were
employed to improve generalization and minimize overfitting. The model was trained and validated
on a diverse dataset of 15 disease classes, achieving high accuracy and robust performance across
crops like tomato, potato, maize, and pepper.
For real-world usability, the trained model was converted to TensorFlow Lite (TFLite) format for
efficient deployment on mobile devices, supporting on-device inference without internet
connectivity. This integration demonstrates the potential of deploying deep learning solutions beyond
laboratory environments, directly benefiting the agricultural community. Overall, the project
showcases a complete deep learning pipeline—from data preprocessing to model compression and
deployment—highlighting the transformative impact of deep learning in precision agriculture.
4
ACKNOWLEDGEMENT
I take this opportunity to express my sincere gratitude and respect to CMR Institute of
Technology, Bengaluru for providing me a platform to pursue my studies and carry out the
Assignement.
It gives me immense pleasure to express my deep sense of gratitude to Dr. Sanjay Jain, Principal,
CMRIT, Bengaluru, for his constant encouragement.
I would like to extend my sincere gratitude to Dr. Shyam P Joy, HOD, Department of Artificial
Intelligence and Machine Learning, CMRIT, Bengaluru, who has been a constant support and
encouragement throughout the course of this project.
I would like to thank my guide Prof. Navya VK, for the valuable guidance throughout the tenure
of this Assignment work.
I would also like to thank all the faculty members of the Artificial Intelligence and Machine
Learning who directly or indirectly encouraged me.
Finally, I thank my parents and friends for all the moral support they have given me during the
completion of this work.
5
GROUP MEMBERS LIST
S. NO. NAME OF STUDENT USN
01 Akshita N Nair 1CR22CI003
02 Harshini R 1CR22CI020
6
TABLE OF CONTENTS
S. No. Contents Page No.
1 Certificate 1
2 Declaration 2
3 Abstract 3
4 Acknowledgement 4
5 Group Members List 5
6 Table of Contents 6
7 Introduction And Objectives 7
8 System Design and Working 9
9 Implementation 13
10 Results and Conclusion 15
7
1. INTRODUCTION AND OBJECTIVES
Agriculture forms the backbone of many economies, yet it remains one of the sectors most vulnerable
to plant diseases, which can severely affect crop yield and food security. Early and accurate detection
of these diseases is critical for ensuring timely intervention and preventing large-scale agricultural loss.
In recent years, deep learning—a branch of artificial intelligence inspired by the human brain—has
emerged as a powerful tool for image-based recognition and classification tasks. Leveraging this
technology, the Crop Care system was developed as a deep learning-based solution for automatic plant
disease detection, enabling real-time diagnosis from leaf images captured using a smartphone camera.
This section presents the background motivation behind the project, the problem being addressed, the
specific objectives of the system, and its overall scope. The focus lies on how deep learning
methodologies, particularly Convolutional Neural Networks (CNNs), can effectively learn to identify
visual symptoms of plant diseases from image data.
1.1. Background and Motivation
Plant diseases are among the leading causes of crop failure worldwide, reducing yield quality and
quantity significantly. Traditional detection methods depend on agricultural experts who visually
inspect plants and identify infections based on their experience. However, such approaches are often
subjective, time-intensive, and inaccessible to farmers in remote or resource-limited regions.
Additionally, environmental factors such as lighting, camera quality, and disease stage can make
manual diagnosis more challenging.
With the rapid advancement of computer vision and deep learning, automated image-based disease
detection has become a viable alternative. Deep learning models, particularly CNNs, can automatically
extract hierarchical patterns from plant images—learning to differentiate between healthy and diseased
leaves without manual feature engineering. This not only improves diagnostic accuracy but also
enables scalable, real-time disease detection that can operate even on mobile devices.
The motivation behind the Crop Care system lies in bridging the gap between cutting-edge AI research
and practical agricultural applications, making intelligent disease detection accessible, affordable, and
efficient.
1.2. Problem Definition
The primary problem addressed in this project is the automatic identification of plant diseases from
digital images of crop leaves using deep learning. The variability in disease symptoms—such as
changes in color, texture, and shape—makes this a challenging computer vision task.
8
The project aims to design and train a CNN capable of classifying plant leaf images into 15 distinct
categories, including both healthy and infected conditions across multiple crop species (tomato, potato,
maize, and pepper).
Key challenges include:
• Handling variations in lighting, orientation, and image quality.
• Preventing overfitting due to limited labeled datasets.
• Achieving high inference speed for real-time mobile deployment.
• Addressing these challenges required a careful balance between model complexity, dataset
diversity, and computational efficiency.
1.3. Objectives
The objectives of the Crop Care deep learning system are as follows:
• To develop and train a CNN-based deep learning model for accurate classification of crop
diseases from leaf images.
• To apply data augmentation and regularization techniques to enhance generalization and
prevent overfitting.
• To evaluate the model’s performance using appropriate metrics such as accuracy and validation
loss, ensuring robust classification across multiple crop types.
• To convert and optimize the trained model into TensorFlow Lite format for efficient mobile
deployment.
• To enable real-time disease diagnosis directly on mobile devices, ensuring accessibility even
in low-connectivity regions.
Through these objectives, the project demonstrates not just theoretical understanding of CNNs but also
practical expertise in building deployable deep learning systems.
9
2. SYSTEM DESIGN AND WORKING
The Crop Care system follows a structured deep learning workflow, beginning from data collection
and preprocessing to model training, evaluation, and mobile deployment. The goal of the system is to
accurately classify plant leaf images into different disease categories by learning discriminative visual
features through a Convolutional Neural Network (CNN).
This section describes the system’s overall design and functionality, covering the dataset preparation,
architecture rationale, training strategy, and deployment workflow.
2.1. System Overview
The system is designed as an end-to-end pipeline, where the CNN model acts as the core decision-
making engine. The process begins when an image of a plant leaf is captured or uploaded by the user.
The image is pre-processed to match the input requirements of the CNN and then passed through the
trained model for inference. The model outputs probabilities for each disease class, and the one with
the highest confidence is selected as the prediction.
Workflow Summary:
1. Image capture/upload by user.
2. Image preprocessing (resize to 256×256, normalization).
3. Forward pass through CNN model (feature extraction and classification).
4. SoftMax layer outputs probability distribution across 15 classes.
5. The class with the highest probability is displayed with its confidence score.
6. On-device inference is achieved through TensorFlow Lite for real-time diagnosis.
This architecture ensures an efficient and scalable design capable of operating both in laboratory testing
and real-world field scenarios.
Fig 1. Workflow Summary
10
2.2. Dataset and Preprocessing
A well-prepared dataset is crucial for effective deep learning model performance. The dataset used in
this project consists of RGB images of crop leaves spanning multiple species (tomato, potato, maize,
and pepper), each annotated with corresponding disease labels. In total, 15 classes are represented,
covering both healthy and diseased samples.
Preprocessing Steps:
• Resizing: All images were resized to 256 × 256 pixels to ensure uniform input dimensions.
• Normalization: Pixel values were rescaled by dividing by 255 to bring them into the [0,1] range,
improving numerical stability during training.
• Augmentation: To improve model generalization and reduce overfitting, several augmentation
techniques were applied:
o Shear Transformation (range = 0.2) to simulate geometric distortion.
o Zoom Transformation (range = 0.2) to create scale variation.
o Horizontal Flipping to mimic real-world leaf orientations.
These augmentations effectively increased dataset diversity and made the model robust to
environmental variations such as lighting, orientation, and background noise.
Input Specifications:
• Image size: 256 × 256 × 3
• Batch size: 16
• Colour space: RGB
• Class mode: Categorical (one-hot encoding for 15 classes)
The preprocessing pipeline ensured that the model received consistent, high-quality inputs for efficient
learning.
2.3. CNN Architecture Design
The backbone of the Crop Care system is a Sequential Convolutional Neural Network (CNN) designed
to automatically learn visual features associated with plant diseases. The network architecture consists
of four convolutional blocks, each followed by a pooling layer, culminating in dense layers for
classification.
Model Architecture:
Input Layer: (256 × 256 × 3)
├─ Conv2D: 32 filters, 3×3 kernel, ReLU activation
├─ MaxPooling2D: 2×2
├─ Conv2D: 64 filters, 3×3 kernel, ReLU activation
├─ MaxPooling2D: 2×2
├─ Conv2D: 128 filters, 3×3 kernel, ReLU activation
11
├─ MaxPooling2D: 2×2
├─ Conv2D: 128 filters, 3×3 kernel, ReLU activation
├─ MaxPooling2D: 2×2
├─ Flatten
├─ Dense: 512 neurons, ReLU activation
├─ Dropout: 0.5
└─ Output: 15 neurons, SoftMax activation
Design Rationale:
• Progressive Feature Extraction: Increasing filters (32→64→128) helps capture simple to
complex visual features, from edges to intricate disease textures.
• MaxPooling Layers: Reduce spatial dimensions by half at each stage, providing translation
invariance and lowering computation.
• Dropout Regularization: A 50% dropout rate prevents overfitting by deactivating random
neurons during training.
• SoftMax Output: Produces a normalized probability distribution across all 15 classes, suitable
for multi-class classification.
This CNN structure offers a strong balance between computational efficiency and classification
accuracy, making it ideal for deployment on constrained mobile hardware.
2.4. Training Configuration
Model training was performed using TensorFlow 2.x with the Keras API, leveraging GPU acceleration
for efficient computation.
Training Parameters:
• Optimizer: Adam (Adaptive Moment Estimation)
o Combines advantages of AdaGrad and RMSProp.
o Provides adaptive learning rates for faster convergence.
• Loss Function: Categorical Cross-Entropy - Measures divergence between predicted and true
class distributions.
• Epochs: Maximum of 50 (early stopping applied).
• Batch Size: 16
• Validation Split: Portion of dataset reserved for validation to monitor overfitting.
Callbacks Used:
• EarlyStopping: Stops training once validation accuracy stops improving.
• ModelCheckpoint: Saves the best-performing model based on validation metrics.
• ReduceLROnPlateau: Reduces learning rate when progress stalls.
12
During training, the model’s accuracy and loss curves were plotted for both training and validation
datasets, allowing visual monitoring of convergence behaviour and generalization performance.
2.5. Model Conversion and Deployment
Once trained, the Keras model was converted to TensorFlow Lite (TFLite) format to enable mobile
inference. This was done using:
Why TensorFlow Lite?
• Optimized for mobile and edge devices.
• Model size reduced by 2–4× through quantization (float32 → int8).
• Faster inference and lower power consumption.
• Enables offline inference — critical for agricultural regions with limited connectivity.
Quantization Impact:
• Slight accuracy trade-off (<1%).
• Significant performance gains in speed and efficiency.
• Lower memory usage, allowing smooth mobile deployment.
While the model is integrated into a lightweight mobile app (built using Flutter), the app primarily
serves as an inference interface — all major computations and disease classification logic are handled
by the deep learning model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
13
3. IMPLEMENTATION
The implementation phase transforms the designed CNN architecture into a functional deep learning
model. It covers dataset loading, model construction, training, tuning, and validation, ensuring that the
system effectively learns to classify plant diseases from images. This section explains the practical
setup and workflow used to train and optimize the Crop Care model.
3.1. Model Development Environment
The model was implemented using TensorFlow 2.x and the Keras API, providing flexibility and
modularity for building neural networks. Training was executed on a GPU-enabled environment to
accelerate computation.
Libraries and Tools Used:
• TensorFlow / Keras: Core deep learning framework.
• NumPy, Pandas: Data handling and preprocessing.
• Matplotlib: Visualization of training curves.
• scikit-learn: Data splitting and evaluation metrics.
This setup ensured a streamlined workflow for experimentation and model tuning.
3.2. Model Training Process
The training pipeline involved preparing the dataset, defining the CNN, and optimizing the model
through iterative learning. The dataset was divided into training and validation subsets (80:20 ratio),
with augmentation applied to increase diversity.
Training Workflow:
1. Feed pre-processed batches of images (256×256×3) into the CNN.
2. Compute categorical cross-entropy loss and update weights using Adam optimizer.
3. Monitor training and validation metrics each epoch.
4. Apply callbacks such as EarlyStopping and ModelCheckpoint for optimal performance
retention.
Training ran for up to 50 epochs, though early stopping typically halted it earlier upon convergence.
The accuracy and loss curves were plotted to visualize performance trends and identify overfitting.
Once the model achieved stable validation accuracy, it was converted to TensorFlow Lite (TFLite)
format for mobile deployment. This conversion reduced model size and latency, enabling efficient on-
device inference without internet connectivity.
14
3.3. Evaluation Metrics
The model’s performance was primarily assessed using classification accuracy, supported by secondary
measures like validation loss and confusion matrix analysis.
• Accuracy: Percentage of correctly predicted samples across all classes.
• Loss: Indicates model’s confidence alignment with true labels.
• Confusion Matrix: Helped identify commonly misclassified disease pairs.
• Learning Curves: Illustrated convergence behaviour across epochs.
High validation accuracy and minimal overfitting confirmed the CNN’s capability to distinguish
diseases effectively, even under variable image conditions.
3.4. Hyperparameter Tuning and Regularization
Optimizing hyperparameters and applying regularization were critical to enhancing model stability.
Key Tuned Parameters:
Parameter Value Purpose
Learning Rate 0.001 Controls weight update magnitude
Batch Size 16 Balances speed and gradient stability
Epochs 50 (with early stop) Defines training cycles
Dropout Rate 0.5 Prevents neuron co-adaptation
Table 1. Parameters
Regularization Techniques:
• Dropout: Randomly deactivates neurons (50%) to prevent overfitting.
• Data Augmentation: Introduces synthetic variations for improved generalization.
• Early Stopping: Automatically halts training once validation performance stabilizes.
Together, these strategies helped the CNN achieve a strong balance between accuracy and
computational efficiency — suitable for real-time, mobile-based plant disease detection.
15
4. RESULTS AND CONLUSION
This section presents the outcomes of the deep learning model after training and testing. It summarizes
the model’s performance in terms of accuracy, validation results, and prediction capability, followed
by a discussion of insights and future improvements. Visual results such as confusion matrices,
accuracy/loss plots, and sample predictions support the findings presented here.
4.1. Experimental Results
The model was trained for up to 50 epochs with early stopping applied to prevent overfitting.
Throughout training, both training and validation accuracy showed a steady upward trend, while the
corresponding losses decreased, confirming effective learning and good generalization.
• Final Training Accuracy: ~98%
• Validation Accuracy: ~95% (average across 15 classes)
• Loss: Converged smoothly with minimal fluctuations
The confusion matrix indicated strong performance across most classes, with only minor
misclassifications in visually similar disease types (e.g., early and late blight in potato).
When tested on unseen real-world leaf samples, the model consistently provided correct classifications
with high confidence (>90% in most cases).
The TensorFlow Lite (TFLite) version of the model maintained near-identical performance with
negligible accuracy drops (<1%), confirming the success of model compression and mobile
optimization.
Fig 2. Model Accuracy
16
Fig 3. Actual vs Predicted
Fig 4. Loss
17
4.2. Model Performance
The trained CNN demonstrated its ability to extract complex spatial features from leaf images,
effectively differentiating disease textures and colour variations. The increasing filter depth (32→128)
across layers helped capture both low-level and high-level features, contributing to its strong
classification accuracy.
The combination of data augmentation and dropout regularization proved highly effective in reducing
overfitting, while the Adam optimizer ensured smooth and adaptive convergence.
On-device testing through the TFLite model showed that predictions were generated in under two
seconds per image on average, enabling real-time plant disease detection. The lightweight architecture
and quantization further allowed efficient inference without the need for internet connectivity,
demonstrating the project’s potential for field-level agricultural use.
4.3. Conclusion and Future Work
The Crop Care system successfully demonstrates the integration of deep learning into agriculture
through automated plant disease detection. By leveraging a custom CNN architecture, the project
achieves high classification accuracy while maintaining computational efficiency suitable for mobile
deployment.
This work highlights how deep learning-based computer vision can empower farmers with accessible,
instant, and reliable disease diagnosis — reducing dependence on manual inspection and improving
timely crop management decisions.
Future Enhancements:
• Expanding the dataset to include more crops and regional disease variations.
• Exploring transfer learning with pre-trained models for faster convergence.
• Implementing explainability techniques such as Grad-CAM to visualize decision regions.
• Extending the mobile app to provide treatment suggestions or connect to agricultural databases.
In conclusion, this project represents a complete deep learning pipeline — from data preprocessing and
CNN training to real-world deployment — underscoring the transformative potential of AI in precision
agriculture.