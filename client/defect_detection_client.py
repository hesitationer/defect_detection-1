"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function


# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from data_util import getModelHelper
import util
from defs import LBLS
import numpy as np
import json
from pprint import pprint
import time
from google.protobuf.json_format import MessageToJson


tf.app.flags.DEFINE_string('server', '104.197.112.172:80',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

class Config:
        """Holdsmodelhyperparamsanddatainformation.

        Theconfigclassisusedtostorevarioushyperparametersanddataset
        informationparameters.ModelobjectsarepassedaConfig()objectat
        instantiation.
        """
        n_channels=3
        x_features=1040
        y_features=780
        n_classes=6
        png_folder='../All_61326/test_61326/'
        test_csv='../test.csv'


def main(_):

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
  val_list = None
  
  config = Config()


  with open(config.test_csv, "r") as val_listfile:
          val_list = util.read_csv(val_listfile)

  print('Time Started')
  start = time.clock()

  helper = getModelHelper(config)

  data = helper.load_and_preprocess_test_data(val_list)
  i = 0
  for dataPoint in data[0]:
          request = predict_pb2.PredictRequest()
          request.model_spec.name = 'predict_defect'
          request.model_spec.signature_name = 'predict_defect'
          request.inputs['image'].CopyFrom(
          tf.contrib.util.make_tensor_proto(dataPoint, shape=np.shape(dataPoint)) )
          result = stub.Predict(request, 100.0)  # 100 secs timeout
          result = MessageToJson(result)
          resultD = json.loads(result)
          floatVal = resultD['outputs']['label']['floatVal']
          index = np.argmax(floatVal)
          print ("File %s --> Predicted %s (Actual %s)"%(val_list[i][0],LBLS[index], LBLS[int(val_list[i][1])]))
          i += 1
  print('Time Taken for requests: ',time.clock() - start)
  return


if __name__ == '__main__':
  tf.app.run()
 

