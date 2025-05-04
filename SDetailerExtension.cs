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
        public static T2IParamGroup Group = new("sDetailer", Toggles: true, Open: false, OrderPriority: -10);
        public static T2IRegisteredParam<string> DetectionModel;
        public static T2IRegisteredParam<float> ConfidenceThreshold;
        public static T2IRegisteredParam<string> MaskFilterMethod;
        public static T2IRegisteredParam<int> MaskTopK;
        public static T2IRegisteredParam<float> MinRatio;
        public static T2IRegisteredParam<float> MaxRatio;
        public static T2IRegisteredParam<int> DilateErode;
        public static T2IRegisteredParam<int> XOffset;
        public static T2IRegisteredParam<int> YOffset;
        public static T2IRegisteredParam<string> MaskMergeInvert;
        public static T2IRegisteredParam<int> MaskBlur;
        public static T2IRegisteredParam<float> DenoisingStrength;
        public static T2IRegisteredParam<string> Prompt;
        public static T2IRegisteredParam<string> NegativePrompt;
        public static T2IRegisteredParam<int> InpaintWidth;
        public static T2IRegisteredParam<int> InpaintHeight;
        public static T2IRegisteredParam<int> Steps;
        public static T2IRegisteredParam<float> CFGScale;
        public static T2IRegisteredParam<T2IModel> Checkpoint;
        public static T2IRegisteredParam<T2IModel> VAE;
        public static T2IRegisteredParam<string> Sampler;
        public static T2IRegisteredParam<long> Seed;
        public static T2IRegisteredParam<string> Scheduler;
        public static T2IRegisteredParam<string> SkipIndices;

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
            string nodeFolder = Path.Join(FilePath, "");
            ComfyUISelfStartBackend.CustomNodePaths.Add(nodeFolder);
            Logs.Init($"Adding {nodeFolder} to CustomNodePaths");
        }

        public override void OnInit()
        {
            ComfyUIBackendExtension.NodeToFeatureMap["SDetailerDetect"] = "comfyui";
            ComfyUIBackendExtension.NodeToFeatureMap["SDetailerInpaintHelper"] = "comfyui";
            DetectionModel = T2IParamTypes.Register<string>(new("SD Detection Model", "Select the detection model used to find objects or faces for inpainting. Requires models in the 'ComfyUI/models/yolov8' folder.",
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

            MaskFilterMethod = T2IParamTypes.Register<string>(new("SD Filter Method", "Method to select which masks to process if multiple are detected: 'area' sorts by size, 'confidence' sorts by detection certainty.", "area",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_filter_method", OrderPriority: 30,
                GetValues: (_) => new List<string> { "area", "confidence" }));

            MaskTopK = T2IParamTypes.Register<int>(new("SD Mask TopK", "Maximum number of detected masks to process, based on the Filter Method and Sort Order (0 = process all).", "0",
                Min: 0, Max: 100, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_topk", OrderPriority: 40));

            MinRatio = T2IParamTypes.Register<float>(new("SD Min Area Ratio", "Ignore masks smaller than this fraction of the total image area (e.g., 0.01 = 1%).", "0.0",
                Min: 0.0f, Max: 1.0f, Step: 0.001f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_min_ratio", OrderPriority: 50));

            MaxRatio = T2IParamTypes.Register<float>(new("SD Max Area Ratio", "Ignore masks larger than this fraction of the total image area (e.g., 0.9 = 90%).", "1.0",
                Min: 0.0f, Max: 1.0f, Step: 0.001f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_max_ratio", OrderPriority: 60));

            SkipIndices = T2IParamTypes.Register<string>(new("SD Skip Indices", "Comma-separated list of detection indices (1-based) to skip *after* sorting. Eg '1,3' skips the first and third detection.", "",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_skip_indices", OrderPriority: 65));

            XOffset = T2IParamTypes.Register<int>(new("SD X Offset", "Shift the generated mask horizontally in pixels (positive = right, negative = left).", "0",
                Min: -200, Max: 200, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_x_offset", OrderPriority: 70));

            YOffset = T2IParamTypes.Register<int>(new("SD Y Offset", "Shift the generated mask vertically in pixels (positive = down, negative = up).", "0",
                Min: -200, Max: 200, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_y_offset", OrderPriority: 80));

            DilateErode = T2IParamTypes.Register<int>(new("SD Dilate/Erode", "Expand (positive value) or shrink (negative value) the mask area by this many pixels. 0 = no change.", "4",
                Min: -128, Max: 128, Step: 4, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_dilate_erode", OrderPriority: 90));

            MaskMergeInvert = T2IParamTypes.Register<string>(new("SD Mask Merge Mode", "How to handle multiple masks: 'none' (process separately), 'merge' (combine into one), 'merge_invert' (combine and invert).", "none",
                Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_merge_invert", OrderPriority: 100,
                GetValues: (_) => new List<string> { "none", "merge", "merge_invert" }));

            MaskBlur = T2IParamTypes.Register<int>(new("SD Mask Blur", "Apply Gaussian blur to the mask edge for smoother inpainting transitions. 0 = sharp edge.", "4",
                Min: 0, Max: 64, Step: 1, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_mask_blur_amount", OrderPriority: 110));

            DenoisingStrength = T2IParamTypes.Register<float>(new("SD Denoising Strength", "Controls how much the original image content is changed within the mask during inpainting (0 = no change, 1 = completely replace).", "0.4",
                Min: 0.0f, Max: 1.0f, Step: 0.05f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_denoising_strength", OrderPriority: 112));

            ConfidenceThreshold = T2IParamTypes.Register<float>(new("SD Confidence Threshold", "Minimum confidence score (0-1) for the detection model to consider an object/face found. Lower values detect more, potentially including false positives.", "0.3",
                Min: 0.05f, Max: 1.0f, Step: 0.05f, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_confidence_threshold", OrderPriority: 115));

            Prompt = T2IParamTypes.Register<string>(new("SD Prompt", "Positive prompt used specifically for the inpainting step. If empty, the main prompt is used.",
                "", Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_prompt", OrderPriority: 120, ViewType: ParamViewType.PROMPT));

            NegativePrompt = T2IParamTypes.Register<string>(new("SD Negative Prompt", "Negative prompt used specifically for the inpainting step. If empty, the main negative prompt is used.",
                "", Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_negative_prompt", OrderPriority: 130, ViewType: ParamViewType.PROMPT));

            Seed = T2IParamTypes.Register<long>(new("SD Seed", "Seed for the inpainting generation. -1 uses a random seed; otherwise, uses the specified value. Does not affect detection.",
                "-1", Min: -1, Max: long.MaxValue, Toggleable: false, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_seed", OrderPriority: 150));

            Checkpoint = T2IParamTypes.Register<T2IModel>(new("SD Checkpoint", "Override Checkpoint: Use a different base model specifically for the inpainting step.",
                null, Toggleable: true, Group: Group, FeatureFlag: "comfyui",
                Subtype: "Stable-Diffusion", ChangeWeight: 9, ID: "sdetailer_checkpoint", OrderPriority: 160,
                GetValues: (session) => Program.T2IModelSets["Stable-Diffusion"].ListModelNamesFor(session)));

            VAE = T2IParamTypes.Register<T2IModel>(new("SD VAE", "Override VAE: Use a different VAE specifically for the inpainting step. 'Automatic' uses the default/checkpoint VAE.",
                null, Toggleable: true, Group: Group,
                FeatureFlag: "comfyui", Subtype: "VAE", ChangeWeight: 7, ID: "sdetailer_vae", OrderPriority: 170,
                GetValues: (session) => Program.T2IModelSets["VAE"].ListModelNamesFor(session)));

            const string NoneSamplerPlaceholder = "(None)";
            Sampler = T2IParamTypes.Register<string>(new("SD Sampler", "Override Sampler: Use a different sampler/scheduler specifically for the inpainting step.",
                NoneSamplerPlaceholder,
                Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_sampler", OrderPriority: 180,
                GetValues: (session) => {
                    T2IParamType samplerType = ComfyUIBackendExtension.SamplerParam?.Type;
                    if (samplerType?.GetValues != null) {
                        try {
                            return [NoneSamplerPlaceholder, .. samplerType.GetValues(session)];
                        }
                        catch { }
                    }
                    return [NoneSamplerPlaceholder];
                }));

            const string NoneSchedulerPlaceholder = "(None)";
            Scheduler = T2IParamTypes.Register<string>(new("SD Scheduler", "Override Scheduler: Use a different scheduler specifically for the inpainting step.",
                NoneSchedulerPlaceholder,
                Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_scheduler", OrderPriority: 185,
                GetValues: (session) => {
                    T2IParamType schedulerType = ComfyUIBackendExtension.SchedulerParam?.Type;
                    if (schedulerType?.GetValues != null) {
                        try {
                            return [NoneSchedulerPlaceholder, .. schedulerType.GetValues(session)];
                        }
                        catch { }
                    }
                    return [NoneSchedulerPlaceholder];
                }));

            Steps = T2IParamTypes.Register<int>(new("SD Steps", "Override Steps: Use a different number of sampling steps specifically for the inpainting step.", "28",
                Min: 1, Max: 150, Step: 1, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_steps", OrderPriority: 190));

            CFGScale = T2IParamTypes.Register<float>(new("SD CFG Scale", "Override CFG Scale: Use a different prompt guidance strength specifically for the inpainting step.", "7.0",
                Min: 0.0f, Max: 30.0f, Step: 0.5f, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_cfg_scale", OrderPriority: 200));

            InpaintWidth = T2IParamTypes.Register<int>(new("SD Width", "Override Width: Set the width for the internal inpainting process. May affect results if different from main width.", "512",
                Min: 64, Max: 2048, Step: 4, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_width", OrderPriority: 210));

            InpaintHeight = T2IParamTypes.Register<int>(new("SD Height", "Override Height: Set the height for the internal inpainting process. May affect results if different from main height.", "512",
                Min: 64, Max: 2048, Step: 4, Toggleable: true, Group: Group, FeatureFlag: "comfyui", ID: "sdetailer_height", OrderPriority: 220));

            WorkflowGenerator.AddStep(g =>
            {
                if (!g.Features.Contains("comfyui"))
                {
                    return;
                }

                if (!g.UserInput.TryGet(DetectionModel, out _))
                {
                    return;
                }

                string detectionModel = g.UserInput.Get(DetectionModel);
                if (detectionModel == "(None)")
                {
                    return;
                }

                float confidenceThreshold = g.UserInput.Get(ConfidenceThreshold, 0.3f);
                string maskFilterMethod = g.UserInput.Get(MaskFilterMethod, "area");
                int maskTopK = g.UserInput.Get(MaskTopK, 0);
                float minRatio = g.UserInput.Get(MinRatio, 0.0f);
                float maxRatio = g.UserInput.Get(MaxRatio, 1.0f);
                int xOffset = g.UserInput.Get(XOffset, 0);
                int yOffset = g.UserInput.Get(YOffset, 0);
                int dilateErode = g.UserInput.Get(DilateErode, 4);
                string maskMergeInvert = g.UserInput.Get(MaskMergeInvert, "none");
                int maskBlur = g.UserInput.Get(MaskBlur, 4);
                float denoisingStrength = g.UserInput.Get(DenoisingStrength, 0.4f);
                string promptText = g.UserInput.Get(Prompt, "");
                string negativePromptText = g.UserInput.Get(NegativePrompt, "");
                int steps = g.UserInput.TryGet(Steps, out int stepsVal) ? stepsVal : g.UserInput.Get(T2IParamTypes.Steps);
                float cfg = g.UserInput.TryGet(CFGScale, out float cfgVal) ? cfgVal : (float)g.UserInput.Get(T2IParamTypes.CFGScale);
                int width = g.UserInput.TryGet(InpaintWidth, out int wVal) ? wVal : g.UserInput.GetImageWidth(512);
                int height = g.UserInput.TryGet(InpaintHeight, out int hVal) ? hVal : g.UserInput.GetImageHeight(512);
                string sortOrder = maskFilterMethod == "area" ? "largest-smallest" : "left-right";

                JArray modelInput = g.FinalModel;
                JArray clipInput = g.FinalClip;
                JArray vaeInput = g.FinalVae;
                string sdLoaderNode = null;
                bool separateVaeUsed = false;

                if (g.UserInput.TryGet(VAE, out T2IModel vaeModel) && vaeModel != null)
                {
                    string vaeLoaderNode = g.CreateNode("VAELoader", new JObject { ["vae_name"] = vaeModel.Name });
                    vaeInput = new JArray { vaeLoaderNode, 0 };
                    separateVaeUsed = true;
                }

                if (g.UserInput.TryGet(Checkpoint, out T2IModel sdModel) && sdModel != null)
                {
                    sdLoaderNode = g.CreateNode("CheckpointLoaderSimple", new JObject { ["ckpt_name"] = sdModel.Name });
                    modelInput = new JArray { sdLoaderNode, 0 };
                    clipInput = new JArray { sdLoaderNode, 1 };
                    if (!separateVaeUsed)
                    {
                        vaeInput = new JArray { sdLoaderNode, 2 };
                    }
                }

                JArray lastNode = g.FinalImageOut;

                string detectNode = g.CreateNode("SDetailerDetect", new JObject()
                {
                    ["image"] = lastNode,
                    ["detection_model"] = detectionModel,
                    ["confidence_threshold"] = confidenceThreshold,
                    ["class_filter"] = "",
                    ["kernel_size"] = dilateErode,
                    ["x_offset"] = xOffset,
                    ["y_offset"] = yOffset,
                    ["mask_merge_mode"] = maskMergeInvert,
                    ["max_detections"] = maskTopK,
                    ["min_ratio"] = minRatio,
                    ["max_ratio"] = maxRatio,
                    ["sort_order"] = sortOrder,
                    ["skip_indices"] = g.UserInput.Get(SkipIndices, "")
                });

                string defaultSampler = g.UserInput.Get(ComfyUIBackendExtension.SamplerParam, "euler");
                string defaultScheduler = g.UserInput.Get(ComfyUIBackendExtension.SchedulerParam, "normal");

                const string NoneSamplerPlaceholder = "(None)";
                const string NoneSchedulerPlaceholder = "(None)";

                string selectedSampler = g.UserInput.TryGet(Sampler, out string samplerVal) ? samplerVal : NoneSamplerPlaceholder;
                string selectedScheduler = g.UserInput.TryGet(Scheduler, out string schedulerVal) ? schedulerVal : NoneSchedulerPlaceholder;

                string finalSampler = (!string.IsNullOrEmpty(selectedSampler) && selectedSampler != NoneSamplerPlaceholder) ? selectedSampler : defaultSampler;
                string finalScheduler = (!string.IsNullOrEmpty(selectedScheduler) && selectedScheduler != NoneSchedulerPlaceholder) ? selectedScheduler : defaultScheduler;

                JArray positiveCond = promptText == "" ? g.FinalPrompt : g.CreateConditioning(promptText, g.FinalClip, g.FinalLoadedModel, true);
                JArray negativeCond = negativePromptText == "" ? g.FinalNegativePrompt : g.CreateConditioning(negativePromptText, g.FinalClip, g.FinalLoadedModel, false);

                string inpaintNode = g.CreateNode("SDetailerInpaintHelper", new JObject()
                {
                    ["image"] = lastNode,
                    ["mask"] = new JArray { detectNode, 1 },
                    ["model"] = modelInput,
                    ["clip"] = clipInput,
                    ["vae"] = vaeInput,
                    ["positive"] = positiveCond,
                    ["negative"] = negativeCond,
                    ["strength"] = denoisingStrength,
                    ["guidance_scale"] = cfg,
                    ["steps"] = steps,
                    ["scheduler"] = finalScheduler,
                    ["sampler_name"] = finalSampler,
                    ["seed"] = g.UserInput.Get(Seed, -1L),
                    ["mask_blur"] = maskBlur,
                });

                g.FinalImageOut = new JArray { inpaintNode, 0 };

            }, 9); 
        }
    }
}
