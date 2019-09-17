"""
Training model for sequence classifier model.
"""


import chainer
import chainer.functions as F
import chainer.links as L


class TrainModel(chainer.Chain):
    """train model."""

    def __init__(self, model, num_candidates=2) -> None:
        super().__init__()
        with self.init_scope():
            self.model = model
            self.fc_img = L.Linear(None, 512)
            self.fc_ctx = L.Linear(None, 512)
            self.fc_cls = L.Linear(None, 1)

        self.num_candidates = num_candidates

    def _mask_loss(self, embs_list, candidates, mask_positions):
        for cand in candidates:
            assert all([len(d) in (0, self.num_candidates) for d in cand])

        imgs = F.concat([F.concat(l, axis=0) for l in candidates], axis=0)
        embs_img = self.model.cnn(imgs)
        embs_ctx = []
        for i, positions in enumerate(mask_positions):
            embs_ctx += [embs_list[i][p] for p in positions]
        embs_ctx = F.stack(embs_ctx)
        embs_ctx = F.repeat(embs_ctx, self.num_candidates, axis=0)
        h_ctx = self.fc_ctx(F.relu(embs_ctx))
        h_img = self.fc_img(F.relu(embs_img))
        h = F.relu(F.concat([h_ctx, h_img]))
        p = self.fc_cls(h).reshape(-1, self.num_candidates)

        gt = self.xp.zeros(len(p), dtype=self.xp.int32)

        mask_loss = F.softmax_cross_entropy(p, gt, ignore_label=-1)
        mask_acc = F.accuracy(p, gt, ignore_label=-1)
        return mask_loss, mask_acc

    def forward(self, images, labels, candidates, mask_positions):
        """Compute loss and metrics.

        Args:
            images (list of ndarray): List of image tensor (L_i, 3, 64, 64),
                where ``L_i`` represents the number of characters in the sequence.
            labels (list of ndarray): List of character label tensor (L_i).
            candidates (list of list of ndarray): List of candidate image list
                for masked character input. Candidate image is ether (0, 3, 64, 64)
                sized array for not-masked input and (4, 3, 64, 64) for 4 candidates
                for the masked character. Example::
                    >>> imgs = np.ones((4, 3, 64, 64), dtype=np.float32)
                    >>> none = np.ones((0, 3, 64, 64), dtype=np.float32)
                    >>> candidates = [[none, imgs, none, none],
                                      [imgs, none, none, imgs]]
                The first item in each candidate array represents the true image
                for the masked position.
            mask_positions (List of np.ndarray): List of array that holds masked
                character position in the sequence.

        Returns:
            chainer.Variable: A variable holding loss data.
        """
        logits, embs_list = self.model(images, True)

        # character classification
        logits_flat = F.concat(logits, axis=0)
        lables_flat = F.concat(labels, axis=0)
        cls_loss = F.softmax_cross_entropy(logits_flat, lables_flat)
        cls_acc = F.accuracy(logits_flat, lables_flat, ignore_label=-1)

        # masked character prediction
        if any([len(pos) for pos in mask_positions]):
            mask_loss, mask_acc = self._mask_loss(embs_list, candidates, mask_positions)
        else:
            mask_loss = chainer.Variable(self.xp.array(0, dtype=self.xp.float32))
            mask_acc = chainer.Variable(self.xp.array(0, dtype=self.xp.float32))

        loss = cls_loss + mask_loss
        chainer.report({'loss': loss,
                        'cls_loss': cls_loss,
                        'cls_acc': cls_acc,
                        'mask_loss': mask_loss,
                        'mask_acc': mask_acc}, self)
        return loss
