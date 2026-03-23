using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class ExecutionProvider
    {
        public string ModelPath { get; set; }


        public ExecutionProvider(string modelPath)
        {
            ModelPath = modelPath;
        }

        protected IYoloDetect BuildInferenceSession(SessionOptions options)
        {
            InferenceSession session = new InferenceSession(ModelPath, options);

            var metaData = session.ModelMetadata.CustomMetadataMap;

            bool isEndToEnd = false;
            if (metaData.ContainsKey("end2end"))
            {
                isEndToEnd = bool.Parse(metaData["end2end"]);
            }

            if (isEndToEnd)
            {
                return new YoloDetectEndToEnd(session, options);
            }
            else
            {
                return new YoloDetectNMS(session, options);
            }
        }
    }
}
