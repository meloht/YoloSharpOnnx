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
        internal IYoloDetect CreateYoloDetect();
        internal IYoloClassify CreateYoloClassify();

        internal IYoloSegment CreateYoloSegment();

        internal IYoloPose CreateYoloPose();

        internal ModelType CurrentModelType { get; }
        internal void SetYoloConfiguration(YoloConfig yoloConfig);
    }
}
