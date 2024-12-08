import unittest


class TestCalculateTiles(unittest.TestCase):
    def setUp(self):
        """初始化测试数据"""
        self.image_width = 1920
        self.image_height = 1080

    def test_exact_match_workers(self):
        """测试 Worker 数量与网格数完全匹配"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 3)
        self.assertEqual(rows * cols, 3)
        self.assertEqual((rows, cols), (1, 3))  # 宽大于高时优先横向切割

    def test_square_preference(self):
        """测试尽量生成接近正方形的网格"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 4)
        self.assertEqual(rows * cols, 4)
        self.assertEqual((rows, cols), (2, 2))  # 4块时优先正方形分割

    def test_large_workers(self):
        """测试较多 Worker 数量"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 12)
        self.assertEqual(rows * cols, 12)
        self.assertEqual((rows, cols), (3, 4))  # 优化为接近宽高比的分割

    def test_one_worker(self):
        """测试单个 Worker 的情况"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 1)
        self.assertEqual(rows * cols, 1)
        self.assertEqual((rows, cols), (1, 1))

    def test_invalid_worker_count(self):
        """测试 Worker 数量为 0 或负数的情况"""
        from common import calculate_grid

        with self.assertRaises(ValueError):
            calculate_grid(self.image_width, self.image_height, 0)

        with self.assertRaises(ValueError):
            calculate_grid(self.image_width, self.image_height, -1)

    def test_aspect_ratio_preservation(self):
        """测试长宽比优先调整"""
        from common import calculate_grid

        rows, cols = calculate_grid(1080, 1920, 4)  # 竖向图片
        self.assertEqual(rows * cols, 4)
        self.assertEqual((rows, cols), (2, 2))  # 竖向也应保持正方形优先

    def test_non_divisible_workers(self):
        """测试当 workers 不能被均匀分割时"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 5)
        self.assertEqual(rows * cols, 5)
        # 检查返回的行列数是否正确
        self.assertTrue(rows == 1 and cols == 5 or rows == 5 and cols == 1)

    def test_large_image_size(self):
        """测试非常大的图像尺寸"""
        from common import calculate_grid

        rows, cols = calculate_grid(8000, 8000, 16)
        self.assertEqual(rows * cols, 16)
        self.assertEqual((rows, cols), (4, 4))  # 优先正方形分割

    def test_wide_image(self):
        """测试宽幅图像"""
        from common import calculate_grid

        rows, cols = calculate_grid(4000, 1000, 8)
        self.assertEqual(rows * cols, 8)
        # 检查是否更倾向于横向分割
        self.assertTrue(cols > rows)

    def test_tall_image(self):
        """测试高幅图像"""
        from common import calculate_grid

        rows, cols = calculate_grid(1000, 4000, 8)
        self.assertEqual(rows * cols, 8)
        # 检查是否更倾向于纵向分割
        self.assertTrue(rows > cols)

    def test_prime_number_workers(self):
        """测试工作者数量为质数的情况"""
        from common import calculate_grid

        rows, cols = calculate_grid(self.image_width, self.image_height, 7)
        self.assertEqual(rows * cols, 7)
        # 检查返回的行列数是否正确
        self.assertTrue(rows == 1 and cols == 7 or rows == 7 and cols == 1)

    def test_zero_image_size(self):
        """测试图像尺寸为 0 的情况"""
        from common import calculate_grid

        with self.assertRaises(ZeroDivisionError):
            calculate_grid(0, self.image_height, 4)
        with self.assertRaises(ZeroDivisionError):
            calculate_grid(self.image_width, 0, 4)


if __name__ == "__main__":
    unittest.main()
