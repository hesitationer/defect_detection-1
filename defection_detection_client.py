# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function


# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from models.data_util import getModelHelper
from models import util
import numpy as np
import json
from pprint import pprint
import time


tf.app.flags.DEFINE_string('server', '10.10.17.221:9003',
                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
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
        png_folder='./data/'


def main(_):

  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
  val_list = None

  with open('./valEqual', "r") as val_listfile:
          val_list = util.read_csv(val_listfile)

  print('Time Started')
  start = time.clock()

  config = Config()

  helper = getModelHelper(config)

  
  data= helper.load_and_preprocess_data(val_list)

  print(np.shape(data)) #, np.shape(label))
  
  for dataPoint in data:
          request = predict_pb2.PredictRequest()
          request.model_spec.name = 'predict_defect'
          request.model_spec.signature_name = 'predict_defect'
          request.inputs['input'].CopyFrom(
                  tf.contrib.util.make_tensor_proto(data, shape=np.shape(data)) )
          result = stub.Predict(request, 100.0)  # 100 secs timeout
          from google.protobuf.json_format import MessageToJson
          result = MessageToJson(result)
          print(result)
          print('Time Taken for 10 requests: ',time.clock() - start)
  return


if __name__ == '__main__':
  tf.app.run()
 

