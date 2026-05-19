using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public sealed class ObjectPool<T> : IDisposable where T : IDisposable
    {
        /// <summary>
        /// 创建对象的方法
        /// </summary>
        private readonly Func<T> _factory;
        private readonly Action<T>? _resetAction;

        /// <summary>
        /// 最大缓存数量
        /// 防止无限增长
        /// </summary>
        private readonly int _maxSize;

        /// <summary>
        /// 使用 ConcurrentBag 保证高性能线程安全
        /// </summary>
        private readonly ConcurrentBag<T> _items = new();


        private bool _disposed;

        public ObjectPool(Func<T> factory, int maxSize = 30, Action<T>? resetAction = null)
        {
            _factory = factory ?? throw new ArgumentNullException(nameof(factory));
            _maxSize = maxSize > 0 ? maxSize : 30;
            _resetAction = resetAction;
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, instance: nameof(ObjectPool<T>));
        }
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            Clear();

            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 获取对象
        /// </summary>
        public T Rent()
        {
            ThrowIfDisposed();
            if (_items.TryTake(out var item))
            {
                return item;
            }

            return _factory();
        }
        /// <summary>
        /// 清空对象池
        /// </summary>
        public void Clear()
        {
            while (_items.TryTake(out var item))
            {
                item?.Dispose();
            }
        }
        /// <summary>
        /// 归还对象
        /// </summary>
        public void Return(T item)
        {
            if (item == null)
                return;

            if (_disposed)
            {
                item?.Dispose();
                return;
            }
            _resetAction?.Invoke(item);
            // 超过最大容量则直接销毁
            if (_items.Count < _maxSize)
            {
                _items.Add(item);
            }
            else
            {
                item?.Dispose();
            }
        }
    }
}
