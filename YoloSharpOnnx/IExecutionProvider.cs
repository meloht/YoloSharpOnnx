using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;

namespace YoloSharpOnnx
{
    public interface IExecutionProvider
    {
        IYoloDetect CreateYoloDetect();
        IYoloClassify CreateYoloClassify();

        IYoloSegment CreateYoloSegment();

        IYoloPose CreateYoloPose();

        ModelType CurrentModelType { get; }
        void SetYoloConfiguration(YoloConfig yoloConfig);
    }
}
