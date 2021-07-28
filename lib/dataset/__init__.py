
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii

from .coco import COCODataset as coco
from .coco import COCOPartDataset as part_coco

from .crowdpose import CrowdPoseDataset as crowd_pose
from .crowdpose import CrowdPosePartDataset as crowd_part