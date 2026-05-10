using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;

namespace YoloSharpOnnx.Inference
{
    public class YoloAsync : IYoloAsync
    {
        private Lazy<IYoloTaskAsync<DetectionResult>> _yoloDetectAsync;
        private Lazy<IYoloTaskAsync<ClsResult>> _yoloClsAsync;

        private IYoloDetect _yoloDetect;
        private IYoloClassify _yoloClassify;
        private YoloConfig _yoloConfig;
        public YoloAsync(IYoloDetect yoloDetect, IYoloClassify yoloClassify,YoloConfig yoloConfig)
        {
            _yoloDetect = yoloDetect;
            _yoloClassify = yoloClassify;
            _yoloConfig = yoloConfig;

            _yoloDetectAsync = new Lazy<IYoloTaskAsync<DetectionResult>>(() => new YoloChannelDetectAsync(_yoloConfig, _yoloDetect.GetYoloProcessAsync(), _yoloDetect.GetRunBatch()));
            _yoloClsAsync = new Lazy<IYoloTaskAsync<ClsResult>>(() => new YoloChannelClsAsync(_yoloConfig, _yoloClassify.GetYoloProcessAsync(), _yoloClassify.GetRunBatch()));
        }
        public void Dispose()
        {
            DisposeLazy(_yoloClsAsync);
            _yoloClsAsync = null;
            DisposeLazy(_yoloDetectAsync);
            _yoloDetectAsync = null;
        }

        public async Task<List<DetectionResult>> RunDetectAsync(string inputImage)
        {
            return await _yoloDetectAsync.Value.RunAsync(inputImage);
        }

        public async Task<List<DetectionResult>> RunDetectAsync(Mat img)
        {
            return await _yoloDetectAsync.Value.RunAsync(img);
        }

        public async Task<List<ClsResult>> RunClassifyAsync(string inputImage)
        {
            return await _yoloClsAsync.Value.RunAsync(inputImage);
        }

        public async Task<List<ClsResult>> RunClassifyAsync(Mat img)
        {
            return await _yoloClsAsync.Value.RunAsync(img);
        }

        private void DisposeLazy<T>(Lazy<T> lazy) where T : IDisposable
        {
            if (lazy != null && lazy.IsValueCreated)
            {
                lazy.Value.Dispose();
            }

        }
    }
}
