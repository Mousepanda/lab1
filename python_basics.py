"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    vowels = set('aeiou')
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0

    # Пока число не станет однозначным
    while number >= 10:
        # Перемножаем цифры числа
        product = 1
        temp_n = number
        while temp_n > 0:
            digit = temp_n % 10
            product *= digit
            temp_n //= 10
        number = product
        count += 1

    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    if n == 0:
        return 0.0 #

    sum_squared_errors = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return sum_squared_errors / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []

    # Деление на 2
    count = 0
    while number % 2 == 0:
        count += 1
        number //= 2
    if count > 0:
        factors.append((2, count))

    # Проверка нечетных делителей от 3 до sqrt(n)
    i = 3
    while i * i <= number:
        count = 0
        while number % i == 0:
            count += 1
            number //= i
        if count > 0:
            factors.append((i, count))
        i += 2

    # Если осталось простое число больше 2
    if number > 2:
        factors.append((number, 1))

    # Формируем строку
    result_parts = []
    for prime, power in factors:
        if power == 1:
            result_parts.append(f"({prime})")
        else:
            result_parts.append(f"({prime}**{power})")

    return "".join(result_parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    current_sum = 0

    # Ищем такое k, что сумма квадратов от 1 до k равна n
    # Сумма квадратов первых k натуральных чисел: k(k+1)(2k+1)/6
    # Но проще просто итеративно добавлять k^2, пока не превысим n

    while True:
        k += 1
        current_sum += k * k
        if current_sum == cube_count:
            return k
        elif current_sum > cube_count:
            return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    length = len(s)

    # Определяем границы левой и правой частей
    # Для четной длины: середина между index length/2 - 1 и length/2
    # Для нечетной: середина index length//2, она не входит никуда

    mid = length // 2

    # Левая часть: от начала до середины (не включая середину, если нечет)
    # Правая часть: от конца назад до середины (не включая середину, если нечет)

    left_part = s[:mid]
    right_part = s[mid + (length % 2):]

    sum_left = sum(int(d) for d in left_part)
    sum_right = sum(int(d) for d in right_part)

    return sum_left == sum_right


# Тесты
if __name__ == "__main__":
    # Тест 1. Подсчёт гласных
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("bcdfg")) == 0

    # Тест 2. Уникальные символы
    assert has_unique_characters(TextInput("abc")) == True
    assert has_unique_characters(TextInput("aabc")) == False

    # Тест 3. Единичные биты
    assert count_one_bits(PositiveIntegerInput(5)) == 2   # 101
    assert count_one_bits(PositiveIntegerInput(7)) == 3   # 111

    # Тест 4. Мультипликативная устойчивость
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

    # Тест 5. Средняя квадратичная ошибка
    assert mse(VectorPairInput([1, 2], [1, 2])) == 0.0
    assert mse(VectorPairInput([1, 2], [2, 3])) == 1.0

    # Тест 6. Разложение на простые множители
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"

    # Тест 7. Пирамида из кубиков
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(30)) == 4
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(10)) == "It is impossible"

    # Тест 8. Сбалансированное число
    assert is_balanced_number(PositiveIntegerInput(1234006)) == True #
    assert is_balanced_number(PositiveIntegerInput(123456)) == False #
    assert is_balanced_number(PositiveIntegerInput(121)) == True     #

    print("Все тесты Части 1 успешно пройдены!")
