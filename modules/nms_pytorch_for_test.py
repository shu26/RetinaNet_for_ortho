import torch

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
class NMS:
    def __init__(self, BBoxTransform, ClipBoxes):
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def calc_from_retinanet_output(self, inputs, regression, classification, anchors):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores>0.05)[0, :, 0]
        # scores_over_thresh = scores

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        transformed_anchors_sqz = torch.squeeze(transformed_anchors, dim=0)
        scores = torch.squeeze(scores, dim=0)
        scores = torch.squeeze(scores, dim=1)
        # print('3 ', scores.shape)
        anchors_nms_idx, _ = self.calcurate(transformed_anchors_sqz, scores)

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


    def entire_nms(self, scores, labels, boxes):
        selected_idx = self.calcurate(boxes, scores)[0]
        scores = scores[selected_idx]
        labels = labels[selected_idx]
        boxes = boxes[selected_idx]

        return scores, labels, boxes


    def calcurate(self, boxes, scores, overlap=0.5, top_k=200):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """

        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w*h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]

        # print(keep.shape)
        # print(count)
        return keep, count
