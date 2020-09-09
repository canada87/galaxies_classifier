from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import mean
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle

class GalaxyDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "elliptic")
        self.add_class("dataset", 2, "circular")
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            if is_train and int(image_id) >= 200:
                continue
            if not is_train and int(image_id) < 200:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id = image_id, path = img_path, annotation = ann_path, class_ids=[0,1,2])#class_ids rappresenta il numero di classi presenti compreso il bkg

    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        class_gal = list()
        for classes in root.findall('.//object'):
            class_gal.append(classes.find('name').text)
        for box in root.findall('.//bndbox'):
            xmin = int(float(box.find('xmin').text))
            ymin = int(float(box.find('ymin').text))
            xmax = int(float(box.find('xmax').text))
            ymax = int(float(box.find('ymax').text))
            # coors = [xmin, ymin, xmax, ymax]
            coors = [ymin, xmin, ymax, xmax]
            boxes.append(coors)
        width = int(root.find('.//size/height').text)
        height = int(root.find('.//size/width').text)
        return boxes, width, height, class_gal

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h, class_gal = self.extract_boxes(path)
        masks = zeros([h,w,len(boxes)], dtype = 'uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if class_gal[i] == 'elliptic':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('elliptic'))
            else:
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('circular'))

        return masks, asarray(class_ids, dtype = 'int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

def plot_actual_vs_predicted(dataset, model, cfg, n_images = 5, train = True):
    for i in range(n_images):
        image = dataset.load_image(i)
        if train:
            mask, _ = dataset.load_mask(i)
        else:
            mask, _ = dataset.load_mask(i+200)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose = 0)[0]
        pyplot.subplot(n_images, 2, i*2+1)
        pyplot.axis('off')
        pyplot.imshow(image)
        if i == 0:
            pyplot.title('Actual')
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:,:,j], cmap = 'gray', alpha = 0.3)
        pyplot.subplot(n_images, 2, i*2+2)
        pyplot.axis('off')
        pyplot.imshow(image)
        if i == 0:
            pyplot.title('Predicted')
        ax = pyplot.gca()
        for box in yhat['rois']:
            y1,x1,y2,x2 = box
            width, height = x2-x1, y2-y1
            rect = Rectangle((x1,y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
    pyplot.show()

def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask = False)
        scale_image = mold_image(image, cfg)
        sample = expand_dims(scale_image, 0)
        yhat = model.detect(sample, verbose = 0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    mAP = mean(APs)
    return mAP

class PredictionConfig(Config):
    NAME = 'galaxy_cfg'
    NUM_CLASSES =  1 + 2
    GPU_CLASSES = 1
    IMAGES_PER_GPU = 1
#
class GalaxyConfig(Config):
    NAME = 'galaxy_cfg'
    NUM_CLASSES =  1 + 2
    STEPS_PER_EPOCH = 200

train_set = GalaxyDataset()
train_set.load_dataset('fake_galaxy', is_train = True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

test_set = GalaxyDataset()
test_set.load_dataset('fake_galaxy', is_train = False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

####### training ########
# config = GalaxyConfig()
# config.display()
# model = MaskRCNN(mode = 'training', model_dir='./', config = config)
# model.keras_model.metrics_tensors = []
# model.load_weights('mask_rcnn_coco.h5', by_name = True, exclude = ['mrcnn_class_logits', 'mrcnn_bbox_fc',"mrcnn_bbox",'mrcnn_mask'])
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs = 5, layers = 'heads')

####### accuracy check #######
# cfg = PredictionConfig()
# model = MaskRCNN(mode = 'inference', model_dir='./', config = cfg)
# model_path = 'mask_rcnn_galaxy_cfg_0005.h5'
# model.load_weights(model_path, by_name = True)
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)

###### detection #########
# cfg = PredictionConfig()
# model = MaskRCNN(mode = 'inference', model_dir='./', config = cfg)
# model_path = 'mask_rcnn_galaxy_cfg_0005.h5'
# model.load_weights(model_path, by_name = True)
# plot_actual_vs_predicted(train_set, model, cfg)
# plot_actual_vs_predicted(test_set, model, cfg)


###### detection mrcnn func #########
cfg = PredictionConfig()
model = MaskRCNN(mode = 'inference', model_dir = '/', config=cfg)
model_path = 'mask_rcnn_galaxy_cfg_0005.h5'
model.load_weights(model_path, by_name = True)
class_names = ['BG','elliptic', 'circular']

# for i in range(200,230):
# img = load_img('./fake_galaxy/images/'+str(i).zfill(4)+'.jpg')
img = load_img('./fake_galaxy/original/0397.jpg')
img = img_to_array(img)
result = model.detect([img], verbose = 0)
r = result[0]
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


# image_id = 0

# image = train_set.load_image(image_id)
# print(image.shape)
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# pyplot.imshow(image)
#
# for j in range(mask.shape[2]):
#     pyplot.imshow(mask[:,:,j], cmap = 'gray', alpha = 0.3)

# pyplot.show()


# for image_id in train_set.image_ids:
#     info = train_set.image_info[image_id]
#     print(info)

# image = train_set.load_image(image_id)
# mask, class_ids = train_set.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image, bbox, mask, class_ids, train_set.class_names)


# image_id = 2
# #
# # image = test_set.load_image(image_id)
# # print(image.shape)
# # mask, class_ids = test_set.load_mask(image_id)
# # print(mask.shape)
# # pyplot.imshow(image)
#
# # for image_id in test_set.image_ids:
# #     info = test_set.image_info[image_id]
# #     print(info)
# #
# image = test_set.load_image(image_id)
# mask, class_ids = test_set.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image, bbox, mask, class_ids, test_set.class_names)
