using SwarmUI.Core;
using SwarmUI.Utils;
using SwarmUI.Text2Image;
using SwarmUI.Builtin_ComfyUIBackend;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.IO;
using System;
using System.Linq;
using SwarmUI.Accounts;
using FreneticUtilities.FreneticExtensions;
using SwarmUI.WebAPI;

namespace SDetailerExtension
{
    public class SDetailerExtension : Extension
    {
        public static T2IParamGroup Group = new("sDetailer", Toggles: true, Open: false, OrderPriority: -2);
        public static T2IRegisteredParam<string> DetectionModel;
        public static T2IRegisteredParam<float> ConfidenceThreshold;
        public static T2IRegisteredParam<string> MaskFilterMethod;
        public static T2IRegisteredParam<string> ClassFilter;
        public static T2IRegisteredParam<int> MaskTopK; // Note: SwarmYoloDetection in SwarmSegWorkflow doesn't directly use TopK. Behavior might change.
        public static T2IRegisteredParam<float> MinRatio; // Note: SwarmYoloDetection in SwarmSegWorkflow doesn't directly use MinRatio.
        public static T2IRegisteredParam<float> MaxRatio; // Note: SwarmYoloDetection in SwarmSegWorkflow doesn't directly use MaxRatio.
        public static T2IRegisteredParam<int> DilateErode;
        public static T2IRegisteredParam<int> XOffset; // Note: SwarmSegWorkflow structure doesn't have a direct XOffset node. This param might not have an effect.
        public static T2IRegisteredParam<int> YOffset; // Note: SwarmSegWorkflow structure doesn't have a direct YOffset node. This param might not have an effect.
        public static T2IRegisteredParam<string> MaskMergeInvert; // Note: SwarmYoloDetection in SwarmSegWorkflow doesn't directly use this.
        public static T2IRegisteredParam<int> MaskBlur;
        public static T2IRegisteredParam<float> DenoisingStrength;
        public static T2IRegisteredParam<string> Prompt;
        public static T2IRegisteredParam<string> NegativePrompt;
        public static T2IRegisteredParam<int> Steps;
        public static T2IRegisteredParam<float> CFGScale;
        public static T2IRegisteredParam<T2IModel> Checkpoint;
        public static T2IRegisteredParam<T2IModel> VAE;
        public static T2IRegisteredParam<string> Sampler;
        public static T2IRegisteredParam<long> Seed;
        public static T2IRegisteredParam<string> Scheduler;
        public static T2IRegisteredParam<string> SkipIndices; // Note: SwarmYoloDetection takes 'filter_by_index' (single int). Mapping might be limited.

        // Helper to get model path string
        private static JToken GetModelPath(T2IModel model, JToken defaultValue)
        {
            if (model == null)
            {
                return defaultValue;
            }
            return model.ToString();
        }

        public override void OnPreInit()
        {
            string nodeFolder = Path.Join(FilePath, "nodes");
            ComfyUISelfStartBackend.CustomNodePaths.Add(nodeFolder);
            Logs.Init($"Adding {nodeFolder} to CustomNodePaths");

            // Register relevant SwarmUI nodes if they are not automatically available
            // This ensures the workflow generator can find them.
            // These are typically built-in or part of SwarmUI's Comfy backend.
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmYoloDetection"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmMaskBlur"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["GrowMask"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmMaskThreshold"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmMaskBounds"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmImageCrop"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["CropMask"] = "comfyui"; // Might be a custom or specific Swarm node
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmImageScaleForMP"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["VAEEncode"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["SetLatentNoiseMask"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["DifferentialDiffusion"] = "comfyui"; // SwarmUI specific
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmKSampler"] = "comfyui"; // SwarmUI KSampler
            ComfyUIBackendExtension.NodeToFeatureMap["VAEDecode"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["ImageScale"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["ThresholdMask"] = "comfyui"; // Core ComfyUI node
            ComfyUIBackendExtension.NodeToFeatureMap["SwarmImageCompositeMaskedColorCorrecting"] = "comfyui";
        }

        public override void OnInit()
        {
            // Keep original SDetailer custom node mappings in case they are used elsewhere or for fallback
            ComfyUIBackendExtension.NodeToFeatureMap["SDetailerDetect"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["SDetailerInpaintHelper"] = "comfyui";

            // Parameter registrations (unchanged from original sDetailer)
            DetectionModel = T2IParamTypes.Register<string>(new("SD Detection Model", "Select detection model for inpainting. Models go in 'SwarmUI/Models/yolov8'.",
                "(None)",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_detection_model", OrderPriority: 10,
                GetValues: (_) => {
                    var models = ComfyUIBackendExtension.YoloModels?.ToList();
                    if (models == null || models.Count == 0) {
                        return ["(None)"];
                    }
                    return models;
                }
            ));

            ClassFilter = T2IParamTypes.Register<string>(new("SD Class Filter", "Comma-separated list of class names or IDs to include (e.g., 'person, car, 3'). Leave empty to include all classes.", "",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_class_filter", OrderPriority: 25));

            MaskFilterMethod = T2IParamTypes.Register<string>(new("SD Filter Method", "Prioritize multiple masks by 'area' (larger first) or 'confidence' (higher score first).", "area",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_filter_method", OrderPriority: 30,
                GetValues: (_) => new List<string> { "area", "confidence" }));

            MaskTopK = T2IParamTypes.Register<int>(new("SD Mask TopK", "Max masks to inpaint, based on 'SD Filter Method'. 0 = process all.", "0",
                Min: 0, Max: 100, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_topk", OrderPriority: 40));

            MinRatio = T2IParamTypes.Register<float>(new("SD Min Area Ratio", "Ignore masks smaller than this fraction of image area (e.g., 0.1 for 10%).", "0.0",
                Min: 0.0f, Max: 1.0f, Step: 0.1f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_min_ratio", OrderPriority: 50));
 
            MaxRatio = T2IParamTypes.Register<float>(new("SD Max Area Ratio", "Ignore masks larger than this fraction of image area (e.g., 0.9 for 90%).", "1.0",
                Min: 0.0f, Max: 1.0f, Step: 0.1f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_max_ratio", OrderPriority: 60));
 
            SkipIndices = T2IParamTypes.Register<string>(new("SD Skip Indices", "Comma-separated mask indices (1-based) to skip after sorting (e.g., '1,3'). Used for 'filter_by_index' in SwarmYoloDetection (takes one int, 0=none).", "",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_skip_indices", OrderPriority: 65));
 
            XOffset = T2IParamTypes.Register<int>(new("SD X Offset", "Shift mask horizontally (pixels). Positive = right, negative = left. (May not apply with SwarmSegWorkflow structure)", "0",
                Min: -200, Max: 200, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_x_offset", OrderPriority: 70));
 
            YOffset = T2IParamTypes.Register<int>(new("SD Y Offset", "Shift mask vertically (pixels). Positive = down, negative = up. (May not apply with SwarmSegWorkflow structure)", "0",
                Min: -200, Max: 200, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_y_offset", OrderPriority: 80));
 
            DilateErode = T2IParamTypes.Register<int>(new("SD Dilate/Erode", "Expand (positive) or shrink (negative) mask area in pixels. 0 = no change. Maps to GrowMask 'expand'.", "4",
                Min: -128, Max: 128, Step: 4, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_dilate_erode", OrderPriority: 90));

            MaskMergeInvert = T2IParamTypes.Register<string>(new("SD Mask Merge Mode", "Handle multiple masks: 'none' (separate), 'merge' (combine), 'merge_invert' (combine & invert). (May not apply with SwarmSegWorkflow structure)", "none",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_merge_invert", OrderPriority: 100,
                GetValues: (_) => new List<string> { "none", "merge", "merge_invert" }));

            MaskBlur = T2IParamTypes.Register<int>(new("SD Mask Blur", "Blur mask edge (pixels) for smoother transitions. 0 = sharp edge. Maps to SwarmMaskBlur 'blur_radius'.", "4",
                Min: 0, Max: 64, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_blur_amount", OrderPriority: 110));

            DenoisingStrength = T2IParamTypes.Register<float>(new("SD Denoising Strength", "How much original image is changed in mask (0 = none, 1 = full replace). For detailer KSampler.", "0.4",
                Min: 0.0f, Max: 1.0f, Step: 0.05f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_denoising_strength", OrderPriority: 112));

            ConfidenceThreshold = T2IParamTypes.Register<float>(new("SD Confidence Threshold", "Min detection score (0-1) to consider an object found. Lower = more detections.", "0.3",
                Min: 0.05f, Max: 1.0f, Step: 0.05f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_confidence_threshold", OrderPriority: 115));

            Prompt = T2IParamTypes.Register<string>(new("SD Prompt", "Positive prompt for inpainting. Uses main prompt if empty.",
                "", Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_prompt", OrderPriority: 120, ViewType: ParamViewType.PROMPT));

            NegativePrompt = T2IParamTypes.Register<string>(new("SD Negative Prompt", "Negative prompt for inpainting. Uses main prompt if empty.",
                "", Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_negative_prompt", OrderPriority: 130, ViewType: ParamViewType.PROMPT));

            Seed = T2IParamTypes.Register<long>(new("SD Seed", "Inpainting seed. -1 for noise consistent with main image seed, 0 for a new random seed for each inpaint, or specify a fixed seed.",
                "-1", Min: -1, Max: long.MaxValue, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_seed", OrderPriority: 150));

            Checkpoint = T2IParamTypes.Register<T2IModel>(new("SD Checkpoint", "Override Checkpoint: Use a different base model for inpainting.",
                null, Toggleable: true, Group: Group, FeatureFlag: "comfyui",
                Subtype: "Stable-Diffusion", ChangeWeight: 9, ID: "sdetailer_checkpoint", OrderPriority: 160,
                GetValues: (session) => {
                    var models = Program.T2IModelSets["Stable-Diffusion"].ListModelNamesFor(session);
                    return models.Where(m => m != "(None)" && m != null)
                                 .Select(m => T2IParamTypes.CleanModelName(m))
                                 .Distinct()
                                 .ToList();
                }));

            VAE = T2IParamTypes.Register<T2IModel>(new("SD VAE", "Override VAE: Use a different VAE for inpainting. 'Automatic' uses default.",
                null, Toggleable: true, Group: Group,
                FeatureFlag: "comfyui", Subtype: "VAE", ChangeWeight: 7, ID: "sdetailer_vae", OrderPriority: 170,
                GetValues: (session) => {
                    var models = Program.T2IModelSets["VAE"].ListModelNamesFor(session);
                    return models.Where(m => m != "(None)" && m != null)
                                 .Select(m => T2IParamTypes.CleanModelName(m))
                                 .Distinct()
                                 .ToList();
                }));

            Sampler = T2IParamTypes.Register<string>(new("SD Sampler", "Override Sampler: Use a different sampler for inpainting.",
                null,
                Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_sampler", OrderPriority: 180,
                GetValues: (session) => {
                    T2IParamType samplerType = ComfyUIBackendExtension.SamplerParam?.Type;
                    if (samplerType?.GetValues != null) {
                        try {
                            return samplerType.GetValues(session);
                        }
                        catch { }
                    }
                    return [];
                }));

            Scheduler = T2IParamTypes.Register<string>(new("SD Scheduler", "Override Scheduler: Use a different scheduler specifically for the inpainting step.",
                null,
                Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_scheduler", OrderPriority: 185,
                GetValues: (session) => {
                    T2IParamType schedulerType = ComfyUIBackendExtension.SchedulerParam?.Type;
                    if (schedulerType?.GetValues != null) {
                        try {
                            return schedulerType.GetValues(session);
                        }
                        catch { }
                    }
                    return [];
                }));

            Steps = T2IParamTypes.Register<int>(new("SD Steps", "Override Steps: Use different sampling steps for inpainting.", "28",
                Min: 1, Max: 150, Step: 1, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_steps", OrderPriority: 190));

            CFGScale = T2IParamTypes.Register<float>(new("SD CFG Scale", "Override CFG Scale: Use different prompt guidance for inpainting.", "7.0",
                Min: 0.0f, Max: 30.0f, Step: 0.5f, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_cfg_scale", OrderPriority: 200));

            WorkflowGenerator.AddStep(g =>
            {
                if (!g.Features.Contains("comfyui"))
                {
                    return;
                }

                if (!g.UserInput.TryGet(DetectionModel, out _)) // Check if sDetailer group is enabled
                {
                    return;
                }

                string detectionModelName = g.UserInput.Get(DetectionModel);
                if (string.IsNullOrEmpty(detectionModelName) || detectionModelName == "(None)")
                {
                    return;
                }

                // Get common parameters
                float confidence = g.UserInput.Get(ConfidenceThreshold, 0.3f);
                string classFilterVal = g.UserInput.Get(ClassFilter, "");
                string maskFilterMethodVal = g.UserInput.Get(MaskFilterMethod, "area");
                string sortOrder = maskFilterMethodVal == "area" ? "largest-smallest" : "left-right"; // As per original sDetailer logic
                
                // For SwarmYoloDetection, filter_by_index is an int.
                // SkipIndices is comma-separated 1-based. For simplicity, we'll use 0 (no specific index filter)
                // or try to parse the first one if provided. SwarmYoloDetection might only take one.
                // The SwarmSegWorkflow.json shows '0' for filter_by_index.
                int filterByIndex = 0;
                string skipIndicesVal = g.UserInput.Get(SkipIndices, "");
                if (!string.IsNullOrWhiteSpace(skipIndicesVal)) {
                    var firstIndex = skipIndicesVal.Split(',')[0].Trim();
                    if (int.TryParse(firstIndex, out int parsedIndex) && parsedIndex > 0) {
                        // filter_by_index in SwarmYoloDetection seems to be 0-based if it's a selection, or a flag.
                        // Given SwarmSegWorkflow uses 0, let's stick to that unless a clear mapping for SkipIndices is defined for SwarmYoloDetection.
                        // For now, we will use the hardcoded 0 from SwarmSegWorkflow.json to match its behavior.
                        // filterByIndex = parsedIndex -1; // If it were 0-based selection
                    }
                }


                JArray initialImage = g.FinalImageOut; // This is the image after the main KSampler and VAEDecode

                // 1. SwarmYoloDetection
                string yoloDetectNode = g.CreateNode("SwarmYoloDetection", new JObject()
                {
                    ["image"] = initialImage,
                    ["model_name"] = detectionModelName,
                    ["confidence"] = confidence,
                    ["filter_by_label"] = classFilterVal,
                    ["sort_order"] = sortOrder, // "left-right" is used in SwarmSegWorkflow.json node 100
                    ["filter_by_index"] = filterByIndex // SwarmSegWorkflow.json uses 0 for node 100
                });
                JArray yoloMaskOutput = new JArray { yoloDetectNode, 0 };

                // 2. SwarmMaskBlur
                int blurRadius = g.UserInput.Get(MaskBlur, 4);
                string maskBlurNode = g.CreateNode("SwarmMaskBlur", new JObject()
                {
                    ["mask"] = yoloMaskOutput,
                    ["blur_radius"] = blurRadius,
                    ["sigma_ratio"] = 1.0f // From SwarmSegWorkflow.json node 101
                });
                JArray blurredMaskOutput = new JArray { maskBlurNode, 0 };

                // 3. GrowMask
                int expand = g.UserInput.Get(DilateErode, 4);
                if (expand < 0) expand = 0; // GrowMask 'expand' should be non-negative
                string growMaskNode = g.CreateNode("GrowMask", new JObject()
                {
                    ["mask"] = blurredMaskOutput,
                    ["expand"] = expand,
                    ["tapered_corners"] = true // From SwarmSegWorkflow.json node 102
                });
                JArray grownMaskOutput = new JArray { growMaskNode, 0 };

                // 4. SwarmMaskThreshold
                string swarmMaskThresholdNode = g.CreateNode("SwarmMaskThreshold", new JObject()
                {
                    ["mask"] = grownMaskOutput,
                    ["min_threshold"] = 0.01f, // From SwarmSegWorkflow.json node 103
                    ["max_threshold"] = 1.0f  // From SwarmSegWorkflow.json node 103
                });
                JArray thresholdedMaskOutput1 = new JArray { swarmMaskThresholdNode, 0 };

                // 5. SwarmMaskBounds
                string maskBoundsNode = g.CreateNode("SwarmMaskBounds", new JObject()
                {
                    ["mask"] = thresholdedMaskOutput1,
                    ["padding"] = 16 // From SwarmSegWorkflow.json node 104
                });
                JArray boundsX = new JArray { maskBoundsNode, 0 };
                JArray boundsY = new JArray { maskBoundsNode, 1 };
                JArray boundsWidth = new JArray { maskBoundsNode, 2 };
                JArray boundsHeight = new JArray { maskBoundsNode, 3 };

                // 6. SwarmImageCrop (on original image)
                string imageCropNode = g.CreateNode("SwarmImageCrop", new JObject()
                {
                    ["image"] = initialImage, // Crop from the output of the main generation
                    ["x"] = boundsX,
                    ["y"] = boundsY,
                    ["width"] = boundsWidth,
                    ["height"] = boundsHeight
                });
                JArray croppedImageOutput = new JArray { imageCropNode, 0 };

                // 7. CropMask (on thresholded mask)
                string maskCropNode = g.CreateNode("CropMask", new JObject()
                {
                    ["mask"] = thresholdedMaskOutput1, // Crop the same mask that was fed to SwarmMaskBounds
                    ["x"] = boundsX,
                    ["y"] = boundsY,
                    ["width"] = boundsWidth,
                    ["height"] = boundsHeight
                });
                JArray croppedMaskOutput = new JArray { maskCropNode, 0 }; // This mask is used for SetLatentNoiseMask and later for ThresholdMask -> Composite

                // 8. SwarmImageScaleForMP (scale cropped image for detailer VAEEncode)
                string imageScaleMPNode = g.CreateNode("SwarmImageScaleForMP", new JObject()
                {
                    ["image"] = croppedImageOutput,
                    ["target_width"] = 1024, // From SwarmSegWorkflow.json node 107
                    ["target_height"] = 1024, // From SwarmSegWorkflow.json node 107
                    ["keep_proportion"] = true // From SwarmSegWorkflow.json node 107
                });
                JArray scaledCroppedImageOutput = new JArray { imageScaleMPNode, 0 };

                // --- Detailer KSampler Pass ---
                JArray detailerModelInput = g.FinalModel;
                JArray detailerClipInput = g.FinalClip;
                JArray detailerVaeInput = g.FinalVae; // VAE for encoding the cropped image & decoding the detailer output

                // Override VAE for detailer pass if specified
                if (g.UserInput.TryGet(VAE, out T2IModel vaeModel) && vaeModel != null)
                {
                    string vaeLoaderNode = g.CreateNode("VAELoader", new JObject { ["vae_name"] = vaeModel.Name });
                    detailerVaeInput = new JArray { vaeLoaderNode, 0 };
                }

                // Override Checkpoint for detailer pass if specified
                if (g.UserInput.TryGet(Checkpoint, out T2IModel sdModel) && sdModel != null)
                {
                    string sdLoaderNode = g.CreateNode("CheckpointLoaderSimple", new JObject { ["ckpt_name"] = sdModel.Name });
                    detailerModelInput = new JArray { sdLoaderNode, 0 };
                    detailerClipInput = new JArray { sdLoaderNode, 1 };
                    // If a new checkpoint is loaded, the VAE might also come from it unless detailerVaeInput was already overridden by SD VAE param
                    if (!(g.UserInput.TryGet(VAE, out T2IModel explicitVaeModel) && explicitVaeModel != null)) {
                         detailerVaeInput = new JArray { sdLoaderNode, 2 };
                    }
                }
                
                // 9. VAEEncode (cropped, scaled image)
                string vaeEncodeNode = g.CreateNode("VAEEncode", new JObject()
                {
                    ["pixels"] = scaledCroppedImageOutput,
                    ["vae"] = detailerVaeInput
                });
                JArray detailerLatentInput = new JArray { vaeEncodeNode, 0 };

                // 10. SetLatentNoiseMask
                string setLatentNoiseMaskNode = g.CreateNode("SetLatentNoiseMask", new JObject()
                {
                    ["samples"] = detailerLatentInput,
                    ["mask"] = croppedMaskOutput // Use the mask cropped by CropMask (node 106 in SwarmSeg)
                });
                JArray maskedLatentForDetailer = new JArray { setLatentNoiseMaskNode, 0 };

                // 11. DifferentialDiffusion (applied to the selected model for detailer)
                string diffDiffusionNode = g.CreateNode("DifferentialDiffusion", new JObject()
                {
                    ["model"] = detailerModelInput
                });
                JArray diffusedModelForDetailer = new JArray { diffDiffusionNode, 0 };

                // Conditioning for Detailer KSampler
                string detailerPromptText = g.UserInput.Get(Prompt, "");
                string detailerNegativePromptText = g.UserInput.Get(NegativePrompt, "");
                JArray detailerPositiveCond = detailerPromptText == "" ? g.FinalPrompt : g.CreateConditioning(detailerPromptText, detailerClipInput, diffusedModelForDetailer, true); // Use detailer's CLIP and Model
                JArray detailerNegativeCond = detailerNegativePromptText == "" ? g.FinalNegativePrompt : g.CreateConditioning(detailerNegativePromptText, detailerClipInput, diffusedModelForDetailer, false); // Use detailer's CLIP and Model

                // Detailer KSampler parameters
                int detailerSteps = g.UserInput.TryGet(Steps, out int stepsVal) ? stepsVal : g.UserInput.Get(T2IParamTypes.Steps);
                float detailerCfg = g.UserInput.TryGet(CFGScale, out float cfgVal) ? cfgVal : (float)g.UserInput.Get(T2IParamTypes.CFGScale);
                string detailerSamplerName = g.UserInput.TryGet(Sampler, out string samplerVal) && samplerVal != null ? samplerVal : g.UserInput.Get(ComfyUIBackendExtension.SamplerParam, "euler");
                string detailerSchedulerName = g.UserInput.TryGet(Scheduler, out string schedulerVal) && schedulerVal != null ? schedulerVal : g.UserInput.Get(ComfyUIBackendExtension.SchedulerParam, "normal");
                long detailerSeed = g.UserInput.Get(Seed, -1L);
                float currentDenoisingStrength = g.UserInput.Get(DenoisingStrength, 0.4f);


                // 12. SwarmKSampler (Detailer Pass)
                // SwarmSegWorkflow.json node 111 uses start_at_step = 10, and total steps = 25.
                // This implies a fixed denoising behavior. If we want to use DenoisingStrength param:
                // int detailer_start_at_step = (int)Math.Round(detailerSteps * (1.0f - currentDenoisingStrength));
                // However, to match SwarmSegWorkflow.json (node 111), we'll use its specific values if DenoisingStrength isn't the primary control there.
                // Node 111 has steps=25, start_at_step=10. Let's use the sDetailer Steps and DenoisingStrength.
                string detailerSamplerNode = g.CreateNode("SwarmKSampler", new JObject()
                {
                    ["model"] = diffusedModelForDetailer,
                    ["positive"] = detailerPositiveCond,
                    ["negative"] = detailerNegativeCond,
                    ["latent_image"] = maskedLatentForDetailer,
                    ["seed"] = detailerSeed,
                    ["steps"] = detailerSteps,
                    ["cfg"] = detailerCfg,
                    ["sampler_name"] = detailerSamplerName,
                    ["scheduler"] = detailerSchedulerName,
                    ["denoise"] = currentDenoisingStrength, // Standard KSampler denoise
                    // SwarmSegWorkflow node 111 specific values that might be part of SwarmKSampler's internal defaults or widgets:
                    ["start_at_step"] = 0, // Standard KSampler usually starts at 0 if denoise is used. Node 111 has 10.
                                           // If SwarmKSampler uses start_at_step for denoise, then:
                                           // ["start_at_step"] = (int)Math.Max(0, detailerSteps - Math.Round(detailerSteps * currentDenoisingStrength)),
                    ["end_at_step"] = 10000, // Default end,
                    ["preview_method"] = "disable", // Common for detailer passes
                    // Other SwarmKSampler params from JSON if needed: "denoise_output", "disable_noise", "default_variations", "width_for_variations"
                });
                JArray detailedLatentOutput = new JArray { detailerSamplerNode, 0 };

                // 13. VAEDecode (detailed latent)
                string vaeDecodeNode2 = g.CreateNode("VAEDecode", new JObject()
                {
                    ["samples"] = detailedLatentOutput,
                    ["vae"] = detailerVaeInput // Use the same VAE as for encoding this pass
                });
                JArray detailedImageOutput = new JArray { vaeDecodeNode2, 0 };

                // 14. ImageScale (scale detailed image back to original crop dimensions)
                string imageScaleBackNode = g.CreateNode("ImageScale", new JObject()
                {
                    ["image"] = detailedImageOutput,
                    ["upscale_method"] = "lanczos", // From SwarmSegWorkflow.json node 113
                    ["width"] = boundsWidth,    // Target original crop width
                    ["height"] = boundsHeight,  // Target original crop height
                    ["crop"] = "disabled"       // From SwarmSegWorkflow.json node 113
                });
                JArray scaledDetailedImageOutput = new JArray { imageScaleBackNode, 0 };

                // 15. ThresholdMask (on the cropped mask from step 7, for compositing)
                // This is node 114 in SwarmSegWorkflow, using the output of CropMask (node 106)
                string thresholdMaskNode2 = g.CreateNode("ThresholdMask", new JObject()
                {
                    ["mask"] = croppedMaskOutput, // Mask from CropMask (node 106)
                    ["threshold"] = 0.0f // From SwarmSegWorkflow.json node 114 (value 0 means everything >0 becomes 1)
                });
                JArray finalCompositeMask = new JArray { thresholdMaskNode2, 0 };
                
                // 16. SwarmImageCompositeMaskedColorCorrecting
                string compositeNode = g.CreateNode("SwarmImageCompositeMaskedColorCorrecting", new JObject()
                {
                    ["destination"] = initialImage, // The original full image before detailing
                    ["source"] = scaledDetailedImageOutput, // The detailed and rescaled patch
                    ["mask"] = finalCompositeMask, // The mask for pasting, derived from croppedMaskOutput
                    ["x"] = boundsX,
                    ["y"] = boundsY,
                    ["correction_method"] = "None" // From SwarmSegWorkflow.json node 115
                });
                JArray finalImageOutput = new JArray { compositeNode, 0 };

                g.FinalImageOut = finalImageOutput;

            }, 9); // Execution priority, 9 is after main sampler typically
        }
    }
}
