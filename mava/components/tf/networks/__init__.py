# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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


from mava.components.tf.networks.additive import AdditiveMixingNetwork
from mava.components.tf.networks.dial import DIALPolicy
from mava.components.tf.networks.epsilon_greedy import epsilon_greedy_action_selector
from mava.components.tf.networks.monotonic import MonotonicMixingNetwork
from mava.components.tf.networks.transformed import QTranAlt, QTranBase
from mava.components.tf.networks.weighted import WeightedMixing
