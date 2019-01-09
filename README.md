# CNN_Facial_Keypoint_Detection
PyTorch CNN for Facial Keypoint Detection

In this project we train a PyTorch CNN to detect facial keypoints.

# Steps:

# 1) Define a CNN in models.py

* Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). 
  The network should take in a grayscale, square image.
  
* Initialize the weights of your CNN by sampling a normal distribution or by performing Xavier initialization 
  so that a particular input signal does not get too big or too small as the network trains (optional).
  
# 2) Define image data transform functions in data_load.py

* Define image transformations to standardize CNN tensor inputs.

* Add a rotation transform to our list of transformations and use it to do data augmentation (optional).

# 3) Notebook 1

* Follow the instructions in the notebook to load and visualize data.

# 4) Notebook 2

* Define a data_transform and apply it whenever you instantiate a DataLoader. 
  The composed transform should include: rescaling/cropping, normalization, and turning input images into torch Tensors. 
  The transform should turn any input image into a normalized, square, grayscale image and then a Tensor for your model to take it as input.

* Select a loss function and optimizer for training the model. 
  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem.

* Train your CNN after defining its loss and optimization functions. 
  You are encouraged, but not required, to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. 
  Save your best trained model.

* After training, all 3 questions about model architecture, choice of loss function, and choice of batch_size and epoch parameters are answered.

* Your CNN "learns" (updates the weights in its convolutional layers) to recognize features and this criteria requires that you 
  extract at least one convolutional filter from your trained model, apply it to an image, and see what effect this filter has on an image.

* After visualizing a feature map, answer: what do you think it detects? This answer should be informed by how a filtered image (from the criteria above) looks.

# 5) Notebook 3

* Use a Haar cascade face detector to detect faces in a given image.

* You should transform any face into a normalized, square, grayscale image and then a Tensor 
  for your model to take in as input (similar to what the data_transform did in Notebook 2).
  
* After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, 
  and display the predicted keypoints for each face in the image.  
  
# 6) Notebook 4

* Have fun with keypoints!

Optional:

* Create face filters that add sunglasses, mustaches, or any .png of your choice to a given face in the correct location.
* Use the keypoints around a person's mouth to estimate the curvature of their mouth and create a smile recognition algorithm .
* Use OpenCV's k-means clustering algorithm to extract the most common facial poses (left, middle, or right-facing, etc.).
* Use the locations of keypoints on two faces to swap those faces.

## If workspace_utils.py is producing errors, simply do not use that module. When calling train_net(n_epochs), remove 'with active_session():' and un-ident the code inside it.
