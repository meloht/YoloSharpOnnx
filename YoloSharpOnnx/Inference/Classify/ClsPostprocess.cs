using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Classify
{
    public class ClsPostprocess : IClsPostprocess
    {
        private readonly OnnxModel _onnxModel;
        private readonly YoloConfig _yoloConfig;
        public ClsPostprocess(OnnxModel onnxModel, YoloConfig yoloConfig)
        {
            _onnxModel = onnxModel;
            _yoloConfig = yoloConfig;
        }
        public List<ClsResult> PostProcess(OrtValue outputValue)
        {
            var arr = outputValue.GetTensorDataAsSpan<float>();
            ClsItem[] res = new ClsItem[arr.Length];

            for (int i = 0; i < arr.Length; i++)
            {
                res[i] = new ClsItem(i, arr[i]);
            }

            Array.Sort(res, (x, y) => y.Value.CompareTo(x.Value));
            List<ClsResult> result = new List<ClsResult>(_yoloConfig.ClassifyTopK);
            for (int i = 0; i < _yoloConfig.ClassifyTopK; i++)
            {
                result.Add(new ClsResult(_onnxModel.Labels[res[i].Index].Name, res[i].Index, res[i].Value));
            }

            return result;
        }

    }
}
