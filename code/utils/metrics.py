OKS_SIGMA = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard iou if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, PIoU=False, PIoU2=False, eps=1e-7, scale=0.0, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or PIoU or PIoU2:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or PIoU or PIoU2 or ShapeIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
            elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif PIoU or PIoU2:
                dw1 = torch.abs(b1_x2.minimum(b1_x1)-b2_x2.minimum(b2_x1))
                dw2 = torch.abs(b1_x2.maximum(b1_x1)-b2_x2.maximum(b2_x1))
                dh1 = torch.abs(b1_y2.minimum(b1_y1)-b2_y2.minimum(b2_y1))
                dh2 = torch.abs(b1_y2.maximum(b1_y1)-b2_y2.maximum(b2_y1))
                P = ((dw1+dw2)/torch.abs(w2)+(dh1+dh2)/torch.abs(h2))/4
                piou_v1 = 1 - iou - torch.exp(-P**2) + 1
                if PIoU:
                    return 1 - piou_v1
                elif PIoU2:
                    q=torch.exp(-P)
                    x=q*Lambda
                    return 1 - 3*x*torch.exp(-x**2)*piou_v1
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        box1, box2 = ops.xyxy2xywh(box1), ops.xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union

def bbox_inner_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, PIoU=False, PIoU2=False, eps=1e-7, ratio=0.7, scale=0.0, Lambda=1.3):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    # IoU
    iou = inter / union

def bbox_mpdiou(box1, box2, xywh=True, mpdiou_hw=2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU

def bbox_inner_mpdiou(box1, box2, xywh=True, mpdiou_hw=2, ratio=0.7, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Inner-IoU
    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return innner_iou - d1 / mpdiou_hw - d2 / mpdiou_hw  # MPDIoU