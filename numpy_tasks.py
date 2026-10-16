"""Заготовки задач на NumPy."""
# numpy_tasks.py
import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = data.matrices # Shape: (p, n, n)
    vectors = data.vectors   # Shape: (p, n, 1)
    result = np.einsum('pij,pjk->ik', matrices, vectors)
    return result

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = data.matrix
    threshold = data.threshold

    # Создаем новую матрицу, где элементы > threshold равны 1, иначе 0
    # Используем astype(int) или можно оставить bool, но обычно ждут числа
    return (matrix > threshold).astype(int)

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for row in matrix:
        unique_vals = np.unique(row)
        result.append(unique_vals.tolist())
    return result

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for col in matrix.T:
        unique_vals = np.unique(col)
        result.append(unique_vals.tolist())
    return result

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed

    # Инициализируем генератор случайных чисел с заданным seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    # Вычисляем статистики
    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    column_vars = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_vars=row_vars,
        column_vars=column_vars
    )

def chess(data: ChessInput) -> np.ndarray:
    m, n, a, b = data.rows, data.columns, data.first, data.second

    # Создаем сетку индексов
    i, j = np.indices((m, n))

    # Если сумма индексов четная, то a, иначе b
    # (0,0) -> 0+0=0 (even) -> a
    matrix = np.where((i + j) % 2 == 0, a, b)
    return matrix

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    h_img, w_img = data.image_height, data.image_width
    rect_h, rect_w = data.height, data.width
    bg_color = np.array(data.background_color)
    shape_color = np.array(data.shape_color)

    # Создаем изображение с фоном
    image = np.tile(bg_color, (h_img, w_img, 1))

    # Вычисляем координаты центра
    cy, cx = h_img // 2, w_img // 2

    # Границы прямоугольника
    y_start = max(0, cy - rect_h // 2)
    y_end = min(h_img, cy + rect_h // 2 + (rect_h % 2))
    x_start = max(0, cx - rect_w // 2)
    x_end = min(w_img, cx + rect_w // 2 + (rect_w % 2))

    # Рисуем прямоугольник
    image[y_start:y_end, x_start:x_end] = shape_color

    return image.astype(np.uint8)

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    h_img, w_img = data.image_height, data.image_width
    a, b = data.semi_axis_x, data.semi_axis_y
    bg_color = np.array(data.background_color)
    shape_color = np.array(data.shape_color)

    image = np.tile(bg_color, (h_img, w_img, 1))

    cy, cx = h_img // 2, w_img // 2

    # Создаем координатную сетку
    y, x = np.ogrid[:h_img, :w_img]

    # Уравнение эллипса: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
    mask = ((x - cx) / a)**2 + ((y - cy) / b)**2 <= 1

    image[mask] = shape_color

    return image.astype(np.uint8)

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.array(data.values)
    window_size = data.window

    mean_val = np.mean(values)
    var_val = np.var(values)
    std_val = np.std(values)

    # Локальные экстремумы
    # Максимум: v[i] > v[i-1] and v[i] > v[i+1]
    # Минимум: v[i] < v[i-1] and v[i] < v[i+1]

    local_max_indices = []
    local_min_indices = []

    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            local_max_indices.append(i)
        elif values[i] < values[i-1] and values[i] < values[i+1]:
            local_min_indices.append(i)

    # Скользящее среднее
    # np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    # mode='valid' дает длину N - M + 1, что совпадает с требованием.
    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')

    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std_deviation=std_val,
        local_maxima_indices=local_max_indices,
        local_minima_indices=local_min_indices,
        moving_average=moving_avg
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.array(data.labels)
    class_count = data.class_count

    if class_count is None:
        class_count = np.max(labels) + 1 if len(labels) > 0 else 0

    n_samples = len(labels)
    one_hot_matrix = np.zeros((n_samples, class_count), dtype=int)

    # Индексация: строка i, столбец labels[i]
    one_hot_matrix[np.arange(n_samples), labels] = 1

    return one_hot_matrix


if __name__ == "__main__":
    # Импорты для тестов
    from grader_contracts.numpy_tasks import (
        BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
        MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
        TimeSeriesInput, TimeSeriesStatistics,
    )

    # Тест 1. Сумма произведений матриц на векторы
    A1 = np.array([[1, 0], [0, 1]])
    A2 = np.array([[0, 1], [1, 0]])
    x1 = np.array([[1], [2]])
    x2 = np.array([[3], [4]])
    # A1*x1 = [[1], [2]], A2*x2 = [[4], [3]] -> Сумма = [[5], [5]]
    res1 = sum_prod(MatrixVectorBatchInput(np.array([A1, A2]), np.array([x1, x2])))
    assert np.allclose(res1, np.array([[5], [5]]))

    # Тест 2. Бинаризация матрицы
    M_bin = np.array([[1, 5, 2], [3, 2, 4]])
    res2 = binarize(BinarizeInput(M_bin, 2))
    assert np.array_equal(res2, np.array([[0, 1, 0], [1, 0, 1]]))

    # Тест 3. Уникальные элементы строк и столбцов
    M_uniq = np.array([[1, 2, 2], [3, 3, 4]])
    # Уникальные в строках (np.unique сортирует)
    assert unique_rows(MatrixInput(M_uniq)) == [[1.0, 2.0], [3.0, 4.0]]
    # Уникальные в столбцах: [1,3], [2,3], [2,4]
    assert unique_columns(MatrixInput(M_uniq)) == [[1.0, 3.0], [2.0, 3.0], [2.0, 4.0]]

    # Тест 4. Статистики случайной матрицы
    res4 = matrix_statistics(RandomMatrixInput(10, 5, 0.0, 1.0, 42))
    assert isinstance(res4, MatrixStatistics)
    assert res4.matrix.shape == (10, 5)
    assert res4.row_means.shape == (10,)
    assert res4.column_means.shape == (5,)

    # Тест 5. Шахматная матрица (пример из условия)
    res5 = chess(ChessInput(3, 4, 0, 1))
    expected5 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(res5, expected5)

    # Тест 6. Прямоугольник и эллипс
    # Прямоугольник 10x10, центр (5,5), размер 4x4
    res_rect = draw_rectangle(RectangleInput(4, 4, 10, 10, (255, 0, 0), (0, 0, 0)))
    assert res_rect.shape == (10, 10, 3)
    assert np.array_equal(res_rect[5, 5], (255, 0, 0)) # Центр закрашен
    assert np.array_equal(res_rect[0, 0], (0, 0, 0))   # Угол - фон

    # Эллипс
    res_ell = draw_ellipse(EllipseInput(2, 2, 10, 10, (0, 255, 0), (0, 0, 0)))
    assert res_ell.shape == (10, 10, 3)
    assert np.array_equal(res_ell[5, 5], (0, 255, 0)) # Центр закрашен

    # Тест 7. Анализ временного ряда
    # Ряд: [1, 3, 2, 4, 1]. Окно: 2
    # Максимумы: индекс 1 (val 3), индекс 3 (val 4)
    # Минимумы: индекс 2 (val 2)
    # Скользящее среднее (окно 2): (1+3)/2=2, (3+2)/2=2.5, (2+4)/2=3, (4+1)/2=2.5
    res7 = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 2))
    assert isinstance(res7, TimeSeriesStatistics)
    assert res7.local_maxima_indices == [1, 3]
    assert res7.local_minima_indices == [2]
    assert np.allclose(res7.moving_average, [2.0, 2.5, 3.0, 2.5])

    # Тест 8. One-hot encoding (пример из условия)
    res8 = one_hot(OneHotInput([0, 2, 3, 0], None))
    expected8 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert np.array_equal(res8, expected8)

    # Граничный случай для One-hot: явное указание class_count
    res8_2 = one_hot(OneHotInput([0, 1], 4))
    assert res8_2.shape == (2, 4)
    assert np.array_equal(res8_2[0], [1, 0, 0, 0])

    print("Все тесты Части 2 (NumPy) успешно пройдены!")
