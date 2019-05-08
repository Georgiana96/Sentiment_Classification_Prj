import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from cnn import LeNet, VGG_16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

current_directory = os.path.dirname(__file__)
SPECTOGRAMS_DIRECTORY = "SPECTOGRAMS/removed_white_border/"

LOAD_MODEL = False
SAVE_MODEL = True

TRAIN_SUBDIR = "train/"
TEST_SUBDIR = "test/"
VALIDATION_SUBDIR = "validation/"

img_rows = 120
img_cols = 160
batch_size = 32
epochs = 30
channels = 3

nb_train_samples = 9117
nb_validation_samples = 1006
nb_test_samples = 2402

print("Loading the training dataset...")
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
            SPECTOGRAMS_DIRECTORY + TRAIN_SUBDIR,
            target_size=(img_cols, img_rows),
            batch_size=batch_size,
            class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                    SPECTOGRAMS_DIRECTORY + VALIDATION_SUBDIR,
                    target_size=(img_cols, img_rows),
                    batch_size=batch_size,
                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
                SPECTOGRAMS_DIRECTORY + TEST_SUBDIR,
                target_size=(img_cols, img_rows),
                batch_size=1,   #change the batch size to 32?
                class_mode='binary',
                shuffle=False)
model = None
if LOAD_MODEL:
    model = load_model('my_model-1.h5')
    print("Loaded model from disk")
else:
    input_shape=(img_cols, img_rows, channels)
    model = VGG_16.build(input_shape, 1)

# opt = SGD(lr=0.001)
opt = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print("Model summary:")
print(model.summary())

history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2)
        # use_multiprocessing=True)

#The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.

##EVALUATE
print("EVALUATE THE MODEL...ON THE VALIDATION SET")
score = model.evaluate_generator(generator=validation_generator,
                         steps=nb_validation_samples // batch_size)

print("Accuracy = " + str(score[1]))
print("Loss = " + str(score[0]))


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()

print("MAKE PREDICTIONS...")
test_generator.reset()
pred = model.predict_generator(test_generator,
                               steps=nb_test_samples // 1,
                               verbose=2)


filenames=test_generator.filenames
print("File names:")
print(filenames)

print("---------OPTION 1-------")
print("Predictions:")
print(pred)
classes = test_generator.classes[test_generator.index_array]
print("Classes:")
print(classes)
pred_class = np.argmax(pred, axis=-1)
print("Pred classes:")
print(pred_class)

print("---------OPTION 2--------")
ground_truth = test_generator.classes
print("Ground truth:")
print(ground_truth)
label2index = test_generator.class_indices
print("Label to index")
print(label2index)
predicted_classes = np.argmax(pred, axis=1)
print("Predicted classes:")
print(predicted_classes)
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))

text_file = open("results-pred-1.txt", "w")
for i in pred:
    text_file.write(str(i))
    text_file.write("\n")
text_file.close()

text_file2 = open("results-names-1.txt", "w")
for i in filenames:
    text_file2.write(str(i))
    text_file2.write("\n")
text_file2.close()


cm = confusion_matrix(test_generator.classes[test_generator.index_array], pred_class)
print("Confusion matrix:")
print(cm)
if SAVE_MODEL:
    model.save('my_model-1.h5')
