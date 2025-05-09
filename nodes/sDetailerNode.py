import math
import torch
import numpy as np
from PIL import Image, ImageChops, ImageDraw
import cv2
import folder_paths
import comfy
import comfy.model_management
import os
# disabled for now
# if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
#     torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

class SDetailerDetect:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            yolo_models = folder_paths.get_filename_list("yolov8")
        except Exception as e:
            print(f"SDetailerDetect: Error getting YOLO models: {e}")
            yolo_models = []

        all_models = yolo_models

        if not all_models:
            print("SDetailerDetect: WARNING - No detection models found in yolov8 folder. Dropdown will be empty.")

        return {
            "required": {
                "image": ("IMAGE",),
                "detection_model": (all_models,),
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "class_filter": ("STRING", {"default": "", "multiline": False}),
                "kernel_size": ("INT", {"default": 4, "min": -64, "max": 64, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "mask_merge_mode": (["none", "merge", "merge_invert"],),
                "max_detections": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                "min_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "max_ratio": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sort_order": (["left-right", "right-left", "top-bottom", "bottom-top", "largest-smallest", "smallest-largest"],),
                "skip_indices": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "sDetailer"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "detect"

    @classmethod
    def IS_CHANGED(cls, detection_model, confidence_threshold, class_filter=None,
                   kernel_size=4, x_offset=0, y_offset=0, mask_merge_mode="none",
                   max_detections=0, min_ratio=0.0, max_ratio=1.0, sort_order="left-right",
                   skip_indices="", **kwargs):
       return (detection_model, confidence_threshold, class_filter, kernel_size,
               x_offset, y_offset, mask_merge_mode, max_detections,
               min_ratio, max_ratio, sort_order, skip_indices)

    def detect(self, image, detection_model, confidence_threshold, class_filter=None,
               kernel_size=4, x_offset=0, y_offset=0, mask_merge_mode="none",
               max_detections=0, min_ratio=0.0, max_ratio=1.0, sort_order="left-right",
               skip_indices=""):

        i = 255.0 * image[0].cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img = img.convert("RGB")
        img_size = img.size
        
        masks = []
        bboxes = []
        confidences = []

        if detection_model == "(None)":
            print("SDetailerDetect: Detection model is (None). Skipping detection.")
            return (image, torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device=image.device))
        
        try:
            model_path = folder_paths.get_full_path("yolov8", detection_model)
            if model_path is None or not os.path.exists(model_path):
                 print(f"SDetailerDetect: Model file '{detection_model}' not found at expected path. Skipping detection.")
                 return (image, torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device=image.device))
        except Exception as e:
             print(f"SDetailerDetect: Error checking model path for '{detection_model}': {e}. Skipping detection.")
             return (image, torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device=image.device))

        mask_output = self.yolo_detect(img, detection_model, confidence_threshold, class_filter)
        masks = mask_output.get("masks", [])
        bboxes = mask_output.get("bboxes", [])
        confidences = mask_output.get("confidences", [])
            
        if not masks:
            print("SDetailerDetect: No initial detections found.")
            return (image, torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32, device=image.device))

        if len(masks) > 0:
            print(f"SDetailerDetect: Sorting {len(masks)} detections by '{sort_order}'")
            sorted_indices = self.sort_masks(masks, sort_order, bboxes)
            sorted_masks = [masks[i] for i in sorted_indices]
            sorted_bboxes = [bboxes[i] for i in sorted_indices]
            sorted_confidences = [confidences[i] for i in sorted_indices]

            skip_set = set()
            if skip_indices and skip_indices.strip():
                try:
                    indices_to_skip = [int(i.strip()) for i in skip_indices.split(',') if i.strip()]
                    skip_set = {idx - 1 for idx in indices_to_skip if 0 < idx <= len(sorted_masks)}
                    print(f"SDetailerDetect: Parsed skip indices (0-based): {skip_set}")
                except ValueError:
                    print(f"SDetailerDetect: Warning - Could not parse skip_indices '{skip_indices}'. Contains non-integer values. Skipping ignored.")

            if skip_set:
                print(f"SDetailerDetect: Applying skipping for indices: {skip_set}")
                filtered_masks_after_skip = []
                filtered_bboxes_after_skip = []
                filtered_confidences_after_skip = []

                for idx, (mask, bbox, conf) in enumerate(zip(sorted_masks, sorted_bboxes, sorted_confidences)):
                    if idx not in skip_set:
                        filtered_masks_after_skip.append(mask)
                        filtered_bboxes_after_skip.append(bbox)
                        filtered_confidences_after_skip.append(conf)
                    else:
                         print(f"SDetailerDetect: Skipping detection at sorted index {idx+1}")

                masks = filtered_masks_after_skip
                bboxes = filtered_bboxes_after_skip
                confidences = filtered_confidences_after_skip
                print(f"SDetailerDetect: {len(masks)} detections remaining after skipping by index.")
            else:
                 masks = sorted_masks
                 bboxes = sorted_bboxes
                 confidences = sorted_confidences
        
        if masks:
             masks = self.mask_preprocess(masks, kernel_size, x_offset, y_offset, mask_merge_mode)
             print(f"SDetailerDetect: {len(masks)} masks after preprocessing.")
        else:
             print("SDetailerDetect: No masks remaining after skipping to preprocess.")


        if len(masks) > 0:
            if min_ratio > 0.0 or max_ratio < 1.0:
                filtered_masks = []
                filtered_bboxes = []
                filtered_confidences = []
                img_area = img_size[0] * img_size[1]
                
                for i, (mask, bbox, conf) in enumerate(zip(masks, bboxes, confidences)):
                    mask_area = torch.sum(mask).item() if isinstance(mask, torch.Tensor) else np.sum(np.array(mask) > 0)
                    ratio = mask_area / img_area
                    if min_ratio <= ratio <= max_ratio:
                        filtered_masks.append(mask)
                        filtered_bboxes.append(bbox)
                        filtered_confidences.append(conf)
                
                masks = filtered_masks
                bboxes = filtered_bboxes
                confidences = filtered_confidences
            
            if max_detections > 0 and len(masks) > max_detections:
                if sort_order == "largest-smallest":
                    areas = [torch.sum(mask).item() if isinstance(mask, torch.Tensor) else np.sum(np.array(mask) > 0) for mask in masks]
                    print(f"SDetailerDetect: Applying max_detections limit of {max_detections} to {len(masks)} masks.")
                    masks = masks[:max_detections]
                    bboxes = bboxes[:max_detections]
                    confidences = confidences[:max_detections]
                    print(f"SDetailerDetect: {len(masks)} masks remaining after max_detections limit.")

        tensor_masks = []
        output_device = image.device
        target_dtype = torch.float32

        for mask in masks:
            mask_tensor = None
            if isinstance(mask, torch.Tensor):
                mask_tensor = mask.to(device=output_device, dtype=target_dtype)
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
                     mask_tensor = mask_tensor[0:1, :, :]
                elif mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
                    print(f"SDetailerDetect: Skipping tensor mask with unexpected shape: {mask.shape}")
                    continue
            elif isinstance(mask, Image.Image):
                try:
                    mask_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_np).to(device=output_device, dtype=target_dtype)
                    mask_tensor = mask_tensor.unsqueeze(0)
                except Exception as e:
                    print(f"SDetailerDetect: Error converting PIL mask to tensor: {e}")
                    continue
            else:
                 print(f"SDetailerDetect: Skipping mask of unknown type: {type(mask)}")
                 continue

            if mask_tensor is not None and (mask_tensor.shape[1] != image.shape[1] or mask_tensor.shape[2] != image.shape[2]):
                mask_tensor = (mask_tensor > 0.5).float()
                
                orig_h, orig_w = mask_tensor.shape[1:]
                target_h, target_w = image.shape[1], image.shape[2]

                if orig_h <= 0 or orig_w <= 0:
                    print(f"SDetailerDetect: Skipping resize for zero-dimension mask ({orig_h}x{orig_w}). Creating empty target mask.")
                    mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                else:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8).squeeze(0)
                    

                    if target_h <= 0 or target_w <= 0:
                        print(f"SDetailerDetect: Invalid target dimensions ({target_h}x{target_w}). Creating empty target mask.")
                        mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(mask_tensor.device)
                    else:
                        if target_h > orig_h or target_w > orig_w:
                            scale_factor = min(target_h / orig_h, target_w / orig_w)
                            
                            if scale_factor <= 0:
                                print(f"SDetailerDetect: Invalid scale_factor ({scale_factor}) calculated. Creating empty target mask.")
                                mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                                mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                            else:
                                calculated_w = round(orig_w * scale_factor)
                                calculated_h = round(orig_h * scale_factor)

                                if calculated_w <= 0 or calculated_h <= 0:
                                    print(f"SDetailerDetect: Calculated fx/fy resize dimensions are invalid ({calculated_w}x{calculated_h}) from scale_factor {scale_factor}. Creating empty target mask.")
                                    mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                                    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                                else:
                                    dsize_large = (calculated_w, calculated_h)

                                    if mask_np is None or mask_np.size == 0:
                                        print(f"SDetailerDetect: mask_np is invalid just before resize. Creating empty target mask.")
                                        mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                                        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                                    elif not math.isfinite(scale_factor):
                                        print(f"SDetailerDetect: scale_factor is not finite ({scale_factor}). Creating empty target mask.")
                                        mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                                        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                                    else:
                                        mask_large = cv2.resize(mask_np, dsize_large, interpolation=cv2.INTER_NEAREST)
                                        
                                        kernel = np.ones((3, 3), np.uint8)
                                        mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel)
                                        
                                        if mask_large is None or mask_large.size == 0:
                                            print(f"SDetailerDetect: Intermediate mask_large became invalid before final resize. Creating empty target mask.")
                                            mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
                                            mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(mask_tensor.device)
                                        else:
                                            mask_np = cv2.resize(mask_large, (target_w, target_h),
                                                               interpolation=cv2.INTER_NEAREST)
                                            
                                            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                                            cleanup_kernel = np.ones((3, 3), np.uint8)
                                            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, cleanup_kernel)
                                            edges = cv2.Canny(mask_np, 100, 200)
                                            edge_kernel = np.ones((3, 3), np.uint8)
                                            edges = cv2.dilate(edges, edge_kernel, iterations=1)
                                            mask_with_edges = cv2.bitwise_or(mask_np, edges)
                                            _, mask_final = cv2.threshold(mask_with_edges, 127, 255, cv2.THRESH_BINARY)
                                            mask_tensor = torch.from_numpy(mask_final.astype(np.float32) / 255.0).to(mask_tensor.device)

                        else:
                            kernel = np.ones((3, 3), np.uint8)
                            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
                            
                            mask_np = cv2.resize(mask_np, (target_w, target_h),
                                               interpolation=cv2.INTER_LANCZOS4)

                            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                            cleanup_kernel = np.ones((3, 3), np.uint8)
                            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, cleanup_kernel)
                            edges = cv2.Canny(mask_np, 100, 200)
                            edge_kernel = np.ones((3, 3), np.uint8)
                            edges = cv2.dilate(edges, edge_kernel, iterations=1)
                            mask_with_edges = cv2.bitwise_or(mask_np, edges)
                            _, mask_final = cv2.threshold(mask_with_edges, 127, 255, cv2.THRESH_BINARY)
                            mask_tensor = torch.from_numpy(mask_final.astype(np.float32) / 255.0).to(mask_tensor.device)

            if mask_tensor is not None and torch.sum(mask_tensor).item() > 0:
                tensor_masks.append(mask_tensor)

        if not tensor_masks:
            print("SDetailerDetect: No valid masks found after processing and filtering. Returning empty mask.")
            final_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=target_dtype, device=output_device)
        elif len(tensor_masks) == 1:
            final_mask = tensor_masks[0]
        else:
            stacked_masks = torch.stack(tensor_masks, dim=0)
            final_mask = torch.max(stacked_masks, dim=0)[0]

        final_mask = final_mask.to(device=output_device, dtype=target_dtype)
        
        return (image, final_mask)
    
    def yolo_detect(self, img, model_name, confidence, class_filter=None):
        model_path = folder_paths.get_full_path("yolov8", model_name)
        if model_path is None or not os.path.exists(model_path):
             print(f"yolo_detect: Model '{model_name}' not found or path invalid. Returning empty.")
             return {"masks": [], "bboxes": [], "confidences": []}
             
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                print(f"Ultralytics import failed. Cannot perform detection.")
                return {"masks": [], "bboxes": [], "confidences": []}
 
            target_device = comfy.model_management.get_torch_device()
 
            model = YOLO(model_path)
 
            selected_classes = None
            if class_filter and class_filter.strip():
                class_filter_list = [cls_name.strip() for cls_name in class_filter.split(",") if cls_name.strip()]
                if hasattr(model, "names"):
                    label_to_id = {name.lower(): id for id, name in model.names.items()}
                    selected_classes = []
                    for cls_name in class_filter_list:
                         if cls_name.isdigit():
                             selected_classes.append(int(cls_name))
                         else:
                             class_id = label_to_id.get(cls_name.lower())
                             if class_id is not None:
                                 selected_classes.append(class_id)
                             else:
                                 print(f"Class '{cls_name}' not found in the model")
             
             
            device_str_for_inference = "" if 'cuda' in str(target_device) else 'cpu'
             
            results = model(img, conf=confidence, device=device_str_for_inference, verbose=False)
 
            if not results or len(results) == 0:
                 return {"masks": [], "bboxes": [], "confidences": []}
             
            boxes = results[0].boxes if results and len(results) > 0 else None
            confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else []
            bboxes_data = boxes.xyxy.cpu().numpy().tolist() if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0 else []
            class_ids = boxes.cls.cpu().numpy().tolist() if boxes is not None and boxes.cls is not None else []

            if results and len(results) > 0 and results[0].masks is not None:
                masks_data = results[0].masks.data
                
                if selected_classes is not None:
                    selected_masks_for_stack = []
                    selected_bboxes_for_return = []
                    selected_confidences_for_return = []
                    
                    for i, class_id_val in enumerate(class_ids):
                        if class_id_val in selected_classes:
                            mask_val = masks_data[i].cpu()
                            selected_masks_for_stack.append(mask_val)
                            selected_bboxes_for_return.append(bboxes_data[i])
                            selected_confidences_for_return.append(confidences[i])
                    
                    masks_data = torch.stack(selected_masks_for_stack) if selected_masks_for_stack else None
                    bboxes_final = selected_bboxes_for_return
                    confidences_final = selected_confidences_for_return
                else:
                   bboxes_final = bboxes_data
                   confidences_final = confidences

                if masks_data is not None and masks_data.shape[0] > 0:
                    pil_masks = []
                    for i_idx in range(masks_data.shape[0]):
                        mask_val_loop = masks_data[i_idx].cpu().float()
                        mask_val_loop = torch.nn.functional.interpolate(
                            mask_val_loop.unsqueeze(0).unsqueeze(0),
                            size=img.size,
                            mode="bilinear"
                        ).squeeze()
                        
                        mask_img = Image.fromarray((mask_val_loop.numpy() * 255).astype(np.uint8))
                        pil_masks.append(mask_img)
                    
                    return {"masks": pil_masks, "bboxes": bboxes_final, "confidences": confidences_final}
                else:
                    print(f"SDetailerDetect: No valid segmentation masks found after processing.")
                    pil_masks = []


            elif bboxes_data:
                 if selected_classes is not None:
                   filtered_bboxes_for_return = []
                   filtered_confidences_for_return = []
                    
                   for i_idx, class_id_val in enumerate(class_ids):
                        if class_id_val in selected_classes:
                            filtered_bboxes_for_return.append(bboxes_data[i_idx])
                            filtered_confidences_for_return.append(confidences[i_idx])
                    
                   bboxes_final_bbox_path = filtered_bboxes_for_return
                   confidences_final_bbox_path = filtered_confidences_for_return
                 else:
                    bboxes_final_bbox_path = bboxes_data
                    confidences_final_bbox_path = confidences

                 pil_masks = []
                 for bbox_val in bboxes_final_bbox_path:
                    mask_img_bbox = Image.new("L", img.size, 0)
                    draw = ImageDraw.Draw(mask_img_bbox)
                    x1, y1, x2, y2 = [int(coord) for coord in bbox_val]
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                    pil_masks.append(mask_img_bbox)

                 return {"masks": pil_masks, "bboxes": bboxes_final_bbox_path, "confidences": confidences_final_bbox_path}
 
 
        except Exception as e:
            print(f"SDetailerDetect: Error during YOLO detection: {e}")
            import traceback
            traceback.print_exc()
        
        return {"masks": [], "bboxes": [], "confidences": []}
    
      
    def mask_preprocess(self, masks, kernel=0, x_offset=0, y_offset=0, mode="none"):
        if not masks:
            return []
        
        if x_offset != 0 or y_offset != 0:
            processed_masks = []
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                    mask_pil = ImageChops.offset(mask_pil, x_offset, -y_offset)
                    mask_np = np.array(mask_pil) / 255.0
                    processed_masks.append(torch.from_numpy(mask_np).float())
                else:
                    processed_masks.append(ImageChops.offset(mask, x_offset, -y_offset))
            masks = processed_masks
        
        if kernel != 0:
            processed_masks = []
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    
                    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                    
                    kernel_size = abs(kernel)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    kernel_matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                    if kernel > 0:
                        processed = cv2.dilate(binary_mask, kernel_matrix, iterations=2 if kernel_size > 5 else 1)
                    else:
                        processed = cv2.erode(binary_mask, kernel_matrix, iterations=1)
                    
                    _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
                    
                    if cv2.countNonZero(processed) > 0:
                        processed_masks.append(torch.from_numpy(processed / 255.0).float())
                else:
                    mask_np = np.array(mask)
                    
                    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                    
                    kernel_size = abs(kernel)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    kernel_matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                    if kernel > 0:
                        processed = cv2.dilate(binary_mask, kernel_matrix, iterations=2 if kernel_size > 5 else 1)
                    else:
                        processed = cv2.erode(binary_mask, kernel_matrix, iterations=1)
                    
                    _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
                    
                    if cv2.countNonZero(processed) > 0:
                        processed_masks.append(Image.fromarray(processed))
            
            masks = processed_masks
        
        if mode != "none" and masks:
            if mode == "merge":
                if isinstance(masks[0], torch.Tensor):
                    merged = masks[0]
                    for m in masks[1:]:
                        merged = torch.maximum(merged, m)
                    masks = [merged]
                else:
                    arrs = [np.array(m) for m in masks]
                    merged = arrs[0]
                    for arr in arrs[1:]:
                        merged = cv2.bitwise_or(merged, arr)
                    masks = [Image.fromarray(merged)]
            
            elif mode == "merge_invert":
                if isinstance(masks[0], torch.Tensor):
                    merged = masks[0]
                    for m in masks[1:]:
                        merged = torch.maximum(merged, m)
                    masks = [1.0 - merged]
                else:
                    arrs = [np.array(m) for m in masks]
                    merged = arrs[0]
                    for arr in arrs[1:]:
                        merged = cv2.bitwise_or(merged, arr)
                    inverted = cv2.bitwise_not(merged)
                    masks = [Image.fromarray(inverted)]
        
        return masks

    def sort_masks(self, masks, sort_order, bboxes):
        num_masks = len(masks)
        if num_masks <= 1:
            return list(range(num_masks))
        
        sort_values = []
        for i, mask in enumerate(masks):
            bbox = bboxes[i] if i < len(bboxes) else None

            if sort_order == "left-right" and bbox:
                val = bbox[0]
            elif sort_order == "right-left" and bbox:
                val = bbox[2]
            elif sort_order == "top-bottom" and bbox:
                val = bbox[1]
            elif sort_order == "bottom-top" and bbox:
                val = bbox[3]
            elif sort_order in ["largest-smallest", "smallest-largest"]:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                val = np.sum(mask_np > 0)
            else:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                val = np.sum(mask_np > 0)

            sort_values.append(val)
        
        indices = np.argsort(sort_values)
        
        if sort_order in ["right-left", "bottom-top", "largest-smallest"]:
            indices = indices[::-1]
        
        return indices
        

NODE_CLASS_MAPPINGS = {
    "SDetailerDetect": SDetailerDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDetailerDetect": "sDetailer Detection",
}
