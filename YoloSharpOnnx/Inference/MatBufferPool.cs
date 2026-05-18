using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public sealed class MatBufferPool : IDisposable
    {
        /// <summary>
        /// 实际缓存池
        /// </summary>
        private readonly ConcurrentBag<ImageBatchData> _pool = new();

        /// <summary>
        /// 最大缓存数量（超过则直接 Dispose）
        /// </summary>
        private readonly int _maxSize;

        /// <summary>
        /// 当前池中缓存数量（近似值）
        /// </summary>
        private int _count;

        /// <summary>
        /// 当前正在使用中的对象数量
        /// </summary>
        private int _usedCount;

        /// <summary>
        /// 用于创建对象
        /// </summary>
        private readonly OnnxModel _onnxModel;

        private bool _disposed;

        public MatBufferPool(int maxSize, OnnxModel onnxModel)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSize);
            _maxSize = maxSize;
            _onnxModel = onnxModel ?? throw new ArgumentNullException(nameof(onnxModel));

            // 预热池
            for (int i = 0; i < _maxSize; i++)
            {
                _pool.Add(new ImageBatchData(_onnxModel));
                _count++;
            }
        }

        /// <summary>
        /// 当前使用中的对象数量
        /// </summary>
        public int UsedCount => Volatile.Read(ref _usedCount);

        /// <summary>
        /// 当前池中缓存数量
        /// </summary>
        public int CachedCount => Volatile.Read(ref _count);

        public ImageBatchData Rent()
        {
            ThrowIfDisposed();

            Interlocked.Increment(ref _usedCount);

            if (_pool.TryTake(out var item))
            {
                Interlocked.Decrement(ref _count);

                // 如果你有 Reset() 方法，建议这里调用
                // item.Reset();

                return item;
            }

            // 池空了，临时创建
            return new ImageBatchData(_onnxModel);

        }
        /// <summary>
        /// 归还对象
        /// </summary>
        public void Return(ImageBatchData item)
        {
            if (item == null)
                return;

            if (_disposed)
            {
                item.Dispose();
                return;
            }

            // 如果你有 Clear() / Reset() 方法，建议这里调用
            // item.Reset();

            Interlocked.Decrement(ref _usedCount);

            // 超过池容量 -> 直接销毁
            if (Interlocked.Increment(ref _count) <= _maxSize)
            {
                _pool.Add(item);
            }
            else
            {
                Interlocked.Decrement(ref _count);
                item.Dispose();
            }
        }

        /// <summary>
        /// 清空池
        /// </summary>
        public void Clear()
        {
            while (_pool.TryTake(out var item))
            {
                item.Dispose();
            }

            Interlocked.Exchange(ref _count, 0);
        }
        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, nameof(MatBufferPool));
        }
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            Clear();

            GC.SuppressFinalize(this);
        }
    }
}
