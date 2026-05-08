using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class OnnxInferenceCore
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        protected readonly FixedBuffer _inputFixedBuffer;
        protected readonly FixedBuffer _outputFixedBuffer;

        protected readonly OnnxModel _onnxModel;
        protected OrtValue _inputOrtValue;
        protected readonly Stopwatch _stopwatch;

        private readonly object _detectLock = new();
        protected MatBufferPool _matPool;
        protected Mat _resizedImg;
        private int _batchPoolSize = 0;
        protected YoloConfig _config;

        public OnnxInferenceCore(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config)
        {
            _config = config;
            _resizedImg = new Mat();
            _onnxModel = onnxModel;
            _stopwatch = new Stopwatch();
            _session = session;
            _options = options;
            _runOptions = new RunOptions();

            _inputFixedBuffer = new FixedBuffer(_onnxModel.InputShapeSize);
            _outputFixedBuffer = new FixedBuffer(_onnxModel.OutputShapeSize);

            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
               _onnxModel.InputShape, _inputFixedBuffer.Address, _onnxModel.InputSizeInBytes);
        }

        public void InitBufferPool(int batchPoolSize)
        {
            if (batchPoolSize != _batchPoolSize)
            {
                lock (_detectLock)
                {
                    if (batchPoolSize != _batchPoolSize)
                    {
                        _matPool?.Dispose();
                        _matPool = null;
                        _batchPoolSize = batchPoolSize;
                    }
                }
            }

            if (_matPool == null)
            {
                lock (_detectLock)
                {
                    if (_matPool == null)
                    {
                        _matPool = new MatBufferPool(batchPoolSize, _onnxModel);
                    }
                }
            }
        }

        public int BufferPoolUsedCount
        {
            get
            {
                if (_matPool == null)
                {
                    return 0;
                }
                return _matPool.UsedCount;
            }
        }

        protected IEnumerable<string[]> GetPreprocessWorkersSize(List<string> listImg)
        {
            int preprocessWorkers = Environment.ProcessorCount;
            if (_onnxModel.DeviceType == DeviceType.CPU)
            {
                preprocessWorkers = 2;
            }
            else
            {
                if (listImg.Count < Environment.ProcessorCount)
                {
                    preprocessWorkers = Environment.ProcessorCount / 2;
                }
                if (listImg.Count < preprocessWorkers)
                {
                    preprocessWorkers = 2;
                }
            }
            int size = listImg.Count / preprocessWorkers;

            if (size < 1)
            {
                size = listImg.Count;
            }
            return listImg.Chunk(size);
        }

        public void DisposeCore()
        {
            _resizedImg.Dispose();
            _matPool?.Dispose();

            _inputFixedBuffer.Dispose();
            _outputFixedBuffer.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
            _options.Dispose();

            _runOptions.Dispose();
            _inputOrtValue.Dispose();
        }
    }
}
