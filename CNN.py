import glob
import numpy as np
import os.path as path
import imageio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_recall_curve, auc,recall_score
from keras.regularizers import l2

# Define image path (e.g.)
IMAGE_PATH = 'D:/Data/HitPeak'
#IMAGE_PATH = 'D:/Data/Peak'
#IMAGE_PATH = 'D:/Data/Hit'

file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))

# Load the images into a single variable and convert to a numpy array
images = [imageio.imread(path) for path in file_paths]
images = np.asarray(images)

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

# Scale images so values are between 0 and 1
images = images / 255

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    if filename[0] == 'W':                          #Every file that begins with W is assigned a 1
        labels[i] = 1
    else:
        labels[i] = 0

# Background = 0 = FALSE
# WIMPS = 1 = TRUE

# Split into test and training sets
TRAIN_TEST_SPLIT = 0.9             

# Split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# Split the images and the labels
x_train = images[train_indices, :, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :, :]
y_test = labels[test_indices]


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=1)


#====================================================================================================

# Convolutional Neural Network

# Hyperparamater - how many convolutional layers the CNN has
N_LAYERS = 2   

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN
    # Define hyperparamters
    MIN_NEURONS = 20
    #MAX_NEURONS = 100
    KERNEL = (3, 3) 
    nuerons = [20,20,20,20]    

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape,kernel_regularizer=l2(0.005)))
            model.add(keras.layers.LeakyReLU(alpha=0.05))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(Conv2D(nuerons[i], KERNEL,kernel_regularizer=l2(0.005)))
            model.add(keras.layers.LeakyReLU(alpha=0.05))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        
    model.add(Dropout(0.25)) 
    model.add(Flatten())
    model.add(Dense(MIN_NEURONS, bias_regularizer=regularizers.l2(0.001),kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.LeakyReLU(alpha=0.05))    
    model.add(Dropout(0.25)) 

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

model = cnn(size=image_size, n_layers=N_LAYERS)

# Training hyperparamters
EPOCHS = 40        
BATCH_SIZE = 100

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto')

# Tensorboard
LOG_DIRECTORY_ROOT = ''
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")    
log_dir = "{}/HitPeak-20000_2cl_40ep_100bs-{}/".format(LOG_DIRECTORY_ROOT, now)  
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, write_grads=True,histogram_freq=1)
callbacks = [early_stopping, tensorboard]

# Plotting/Printing Results
Fit = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1 , validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

accuracy = Fit.history['acc']
val_accuracy = Fit.history['val_acc']
loss = Fit.history['loss']
val_loss = Fit.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training acuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
#plt.savefig('Accuracy.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.savefig('Loss.png')
plt.show()

# Make a prediction on the test set
test_predictions = model.predict(x_test)
#print (test_predictions)
#test_predictions = np.round(test_predictions)

# Report the accuracy
accuracy = accuracy_score(y_test, np.round(test_predictions))
print("Accuracy: " + str(accuracy))
#f1 = f1_score(y_test, np.round(test_predictions))
#print("F1 score: " + str(f1))
average_precision = average_precision_score(y_test, test_predictions)
print("Average precision: " + str(average_precision))

precision, recall, thresholds = precision_recall_curve(y_test, test_predictions)
auc = auc(recall, precision)
recall1 = recall_score(y_test, np.round(test_predictions))
print("recall: " + str(recall1))
print('AUC:' +str(auc))

#Report Confusion Matrix
y_actu = pd.Series(y_test.ravel(), name='Actual')
y_pred = pd.Series(np.round(test_predictions.ravel()), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

#Plot confusion matrix
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap='YlGn'):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0,len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    for i in range(len(df_confusion.index)):
        for j in range(len(df_confusion.columns)):
            plt.text(j,i,str(df_confusion.iloc[i,j]))
    plt.show()

plot_confusion_matrix(df_confusion)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='darkblue', label='AUC = %0.2f' %(auc))
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    #plt.savefig('F:/NewData/20000_75x75/Results/Layers/Same/ROCCurve.png')
    
fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
np.savetxt("F:NewData/ROCCurve/tprHitPeak.txt",tpr)
np.savetxt("F:NewData/ROCCurve/fprHitPeak.txt",fpr)
plot_roc_curve(fpr, tpr)
