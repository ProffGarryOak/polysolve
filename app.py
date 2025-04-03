from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objs as go
import random
app = Flask(__name__)


def format_equation(coefficients):
    try:
        degree = len(coefficients) - 1

        def term(coef, exp):
            if int(coef) == coef:
                coef = int(coef)
            if exp == 0:
                return str(coef)
            elif exp == 1:
                return f"{coef}x"
            else:
                return f"{coef}x^{exp}"

        def superscript(exp):
            superscript_chars = {"0": "⁰", "1": "¹", "2": "²", "3": "³",
                                 "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
            return "".join(superscript_chars[digit] for digit in str(exp))

        terms = [term(coef, exp) for exp, coef in enumerate(
            reversed(coefficients)) if coef != 0]

        if not terms:
            return "0"
        else:
            formatted_equation = " + ".join(terms[::-1])
            for exp in range(degree, 0, -1):
                formatted_equation = formatted_equation.replace(
                    f"^{exp}", superscript(exp))
            return formatted_equation
    except Exception as e:
        return f"Error: {e}"


def format_imaginary_part(imaginary_part):
    try:
        if imaginary_part == 1:
            return "i"
        elif imaginary_part == -1:
            return "-i"
        elif imaginary_part.is_integer():
            return f"{int(imaginary_part)}i"
        else:
            formatted_imaginary = f"{imaginary_part:.5f}"
            formatted_imaginary = formatted_imaginary.rstrip('0').rstrip(
                '.') if '.' in formatted_imaginary else formatted_imaginary
            return f"{formatted_imaginary}i"
    except Exception as e:
        return f"Error: {e}"


def find_root(coefficients, degree):
    try:
        roots = np.roots(coefficients)
        real_parts = np.real(roots)
        imaginary_parts = np.imag(roots)
        ans = []
        l = len(roots)
        for i in range(l):
            if imaginary_parts[i] == 0:
                if real_parts[i].is_integer():
                    ans.append(str(int(real_parts[i])))
                else:
                    formatted_real = f"{real_parts[i]:.5f}"
                    formatted_real = formatted_real.rstrip('0').rstrip(
                        '.') if '.' in formatted_real else formatted_real
                    ans.append(formatted_real)
            elif real_parts[i] == 0:
                ans.append(format_imaginary_part(imaginary_parts[i]))
            else:
                if real_parts[i].is_integer():
                    real_part_str = str(int(real_parts[i]))
                else:
                    formatted_real = f"{real_parts[i]:.5f}"
                    formatted_real = formatted_real.rstrip('0').rstrip(
                        '.') if '.' in formatted_real else formatted_real
                    real_part_str = formatted_real
                imaginary_part_str = format_imaginary_part(imaginary_parts[i])
                ans.append(f"{real_part_str} + {imaginary_part_str}")
        goodlook = " <br><br> ".join(ans)
        return goodlook
    except Exception as e:
        return f"Error: {e}"


def polynomial_addition(poly1, poly2):
    try:
        result = np.polyadd(poly1, poly2)
        return format_equation(result)
    except Exception as e:
        return f"Error: {e}"


def polynomial_multiplication(poly1, poly2):
    try:
        result = np.polymul(poly1, poly2)
        return format_equation(result)
    except Exception as e:
        return f"Error: {e}"


def polynomial_division(poly1, poly2):
    try:
        quotient, remainder = np.polydiv(poly1, poly2)
        quotient_eq = format_equation(quotient)
        remainder_eq = format_equation(remainder)
        return quotient_eq, remainder_eq
    except Exception as e:
        return f"Error: {e}", None


def polynomial_subtraction(poly1, poly2):
    try:
        result = np.polysub(poly1, poly2)
        return format_equation(result)
    except Exception as e:
        return f"Error: {e}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot_graph', methods=['GET', 'POST'])
def plot_graph():
    if request.method == 'POST':
        try:
            coefficients = list(
                map(float, request.form.get('coefficients').split(',')))
            x_values = np.linspace(-10, 10, 100)
            y_values = np.polyval(coefficients, x_values)
            trace = go.Scatter(x=x_values, y=y_values,
                               mode='lines', name='Polynomial Graph', line=dict(color="#071e3c"))
            layout = go.Layout(
                xaxis=dict(title='X', color='black',
                           showgrid=True, gridcolor='grey'),
                yaxis=dict(title='Y', color='black',
                           showgrid=True, gridcolor='grey'),
                paper_bgcolor='white',
                plot_bgcolor="#daeee9",
                font=dict(color='black'),
                template='plotly_dark',
                margin=dict(l=10, r=50, t=50, b=50)
            )
            fig = go.Figure(data=[trace], layout=layout)
            graph = fig.to_html(
                full_html=False, default_height=500, default_width=600)
            return render_template('plot_graph.html', graph=graph)
        except Exception as e:
            return render_template('plot_graph.html', error="Invalid Input")
    return render_template('plot_graph.html')


@app.route('/find_root', methods=['GET', 'POST'])
def find_root_page():
    roots = None
    poly_eq = None
    if request.method == 'POST':
        try:
            coefficients = list(
                map(float, request.form.get('coefficients').split(',')))
            degree = len(coefficients)-1
            roots = find_root(coefficients, degree)
            poly_eq = format_equation(coefficients)
        except Exception as e:
            return render_template('find_root.html', error="Invalid Input")
    return render_template('find_root.html', roots=roots, poly1=poly_eq)


@app.route('/addition', methods=['GET', 'POST'])
def addition_page():
    result = None
    poly1_eq = None
    poly2_eq = None
    if request.method == 'POST':
        try:
            poly1_eq = format_equation(
                list(map(float, request.form.get('poly1').split(','))))
            poly2_eq = format_equation(
                list(map(float, request.form.get('poly2').split(','))))
            poly1 = list(map(float, request.form.get('poly1').split(',')))
            poly2 = list(map(float, request.form.get('poly2').split(',')))
            result = polynomial_addition(poly1, poly2)
        except Exception as e:
            return render_template('addition.html', error="Invalid Input")
    return render_template('addition.html', result=result, poly1=poly1_eq, poly2=poly2_eq)


@app.route('/multiplication', methods=['GET', 'POST'])
def multiplication_page():
    result = None
    poly1_eq = None
    poly2_eq = None
    if request.method == 'POST':
        try:
            poly1_eq = format_equation(
                list(map(float, request.form.get('poly1').split(','))))
            poly2_eq = format_equation(
                list(map(float, request.form.get('poly2').split(','))))
            poly1 = list(map(float, request.form.get('poly1').split(',')))
            poly2 = list(map(float, request.form.get('poly2').split(',')))
            result = polynomial_multiplication(poly1, poly2)
        except Exception as e:
            return render_template('multiplication.html', error="Invalid Input")
    return render_template('multiplication.html', result=result, poly1=poly1_eq, poly2=poly2_eq)


@app.route('/division', methods=['GET', 'POST'])
def division_page():
    quotient = None
    remainder = None
    poly1_eq = None
    poly2_eq = None
    if request.method == 'POST':
        try:
            poly1_eq = format_equation(
                list(map(float, request.form.get('poly1').split(','))))
            poly2_eq = format_equation(
                list(map(float, request.form.get('poly2').split(','))))
            poly1 = list(map(float, request.form.get('poly1').split(',')))
            poly2 = list(map(float, request.form.get('poly2').split(',')))
            quotient, remainder = polynomial_division(poly1, poly2)
        except Exception as e:
            return render_template('division.html', error="Invalid Input")
    return render_template('division.html', quotient=quotient, remainder=remainder, poly1=poly1_eq, poly2=poly2_eq)


@app.route('/subtraction', methods=['GET', 'POST'])
def subtraction_page():
    result = None
    poly1_eq = None
    poly2_eq = None
    if request.method == 'POST':
        try:
            poly1_eq = format_equation(
                list(map(float, request.form.get('poly1').split(','))))
            poly2_eq = format_equation(
                list(map(float, request.form.get('poly2').split(','))))
            poly1 = list(map(float, request.form.get('poly1').split(',')))
            poly2 = list(map(float, request.form.get('poly2').split(',')))
            result = polynomial_subtraction(poly1, poly2)
        except Exception as e:
            return render_template('subtraction.html', error="Invalid Input")
    return render_template('subtraction.html', result=result, poly1=poly1_eq, poly2=poly2_eq)




def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num or board[(row // 3) * 3 + i // 3][(col // 3) * 3 + i % 3] == num:
            return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

@app.route('/sudoku', methods=['GET', 'POST'])
def sudoku_solver():
    board = [[None] * 9 for _ in range(9)]  # Initialize with None values
    solved_board = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'solve':
            try:
                # Read inputs and convert to a 9x9 grid of integers, treating empty inputs as 0
                board = [
                    [int(request.form.get(f'cell-{r}-{c}', '0').strip()) if request.form.get(f'cell-{r}-{c}', '0').strip() else 0
                     for c in range(9)]
                    for r in range(9)
                ]
                # Copy board for solving
                solved_board = [row[:] for row in board]
                if not solve_sudoku(solved_board):
                    return render_template('sudokusolver.html', error="No solution exists", board=board)
            except ValueError:
                return render_template('sudokusolver.html', error="Invalid Input", board=board)
        elif action == 'reset':
            board = [[None] * 9 for _ in range(9)]  # Reset board to initial state
    return render_template('sudokusolver.html', board=board, solved_board=solved_board)


# Initialize the board
def initialize_board():
    return [['' for _ in range(3)] for _ in range(3)]

# Check for a winner or a tie
def check_winner(board):
    # Check rows, columns, and diagonals
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != '':
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '':
        return board[0][2]
    
    # Check for a tie
    for row in board:
        if '' in row:
            return None  # Game is still ongoing
    return 'Tie'

# Minimax Algorithm for the unbeatable AI
def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result == 'X':
        return -1
    elif result == 'O':
        return 1
    elif result == 'Tie':
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score

# Find the best move for the AI
def find_best_move(board):
    best_score = -float('inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ''
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    return best_move

@app.route('/tictactoe', methods=['GET', 'POST'])
def tictactoe_solver():
    if request.method == 'POST':
        data = request.json
        board = data['board']
        best_move = find_best_move(board)
        if best_move:
            board[best_move[0]][best_move[1]] = 'O'
        winner = check_winner(board)
        return jsonify({'board': board, 'winner': winner})
    return render_template('tictactoe.html')

def minimize_transactions(expenses):
    # Calculate total amount spent by each person and their net balance
    total_expenses = sum(amount for _, amount in expenses)
    n = len(expenses)
    avg_expense = total_expenses / n

    # Calculate net balance for each person
    net_balance = {person: amount - avg_expense for person, amount in expenses}

    # Separate creditors and debtors
    creditors = []
    debtors = []
    for person, balance in net_balance.items():
        if balance > 0:
            creditors.append((person, round(balance, 2)))  # Round to avoid floating-point issues
        elif balance < 0:
            debtors.append((person, round(-balance, 2)))  # Round to avoid floating-point issues

    # Minimize transactions by balancing creditors and debtors
    transactions = []
    while creditors and debtors:
        creditor, credit = creditors.pop()
        debtor, debt = debtors.pop()

        # Determine the amount to settle between the two
        settlement = min(credit, debt)
        transactions.append(f"{debtor} pays {creditor} ${settlement:.2f}")

        # Update remaining balances
        credit -= settlement
        debt -= settlement

        # Reinsert remaining creditors and debtors if any balance is left
        if credit > 0:
            creditors.append((creditor, credit))
            creditors.sort(key=lambda x: x[1], reverse=True)  # Re-sort creditors
        if debt > 0:
            debtors.append((debtor, debt))
            debtors.sort(key=lambda x: x[1], reverse=True)  # Re-sort debtors

    return transactions

# Example usage:

@app.route('/split', methods=['GET', 'POST'])
def cash_flow_optimizer():
    if request.method == 'POST':
        data = request.get_json()  # Correctly receive JSON data from the frontend
        expenses = data['expenses']
        transactions = minimize_transactions(expenses)
        return jsonify({'transactions': transactions})
    return render_template('split.html')


def load_word_list():
    with open('static/wordle.txt', 'r') as file:
        words = [word.strip() for word in file.readlines()]
    return words

# Filter word list based on the user's choices
# Global variable to keep track of gray letters
# Global variable to keep track of gray letters
# Declare gray_letters globally
gray_letters = set()

def filter_words(words, guess, feedback):
    global gray_letters
    filtered_words = []
    guess = guess.lower()

    # Track green and yellow letters to prevent incorrectly marking gray letters
    green_yellow_letters = set()
    letter_count = {}  # Track letter counts for guess

    # Track gray letters from the current guess and feedback
    for i in range(5):
        if feedback[i] == "green" or feedback[i] == "yellow":
            green_yellow_letters.add(guess[i].lower())
            # Count the number of valid occurrences of each letter
            letter_count[guess[i].lower()] = letter_count.get(guess[i].lower(), 0) + 1

    # Add letters to gray if they are not in green/yellow
    for i in range(5):
        if feedback[i] == "gray" and guess[i].lower() not in green_yellow_letters:
            gray_letters.add(guess[i].lower())

    for word in words:
        word = word.lower()

        # Count occurrences of each letter in the word
        word_letter_count = {}
        for char in word:
            word_letter_count[char] = word_letter_count.get(char, 0) + 1

        # Check if the word contains any gray letters
        if any(letter in gray_letters for letter in word):
            continue

        match = True
        used_indices = set()

        # Check for correct counts of green/yellow letters
        for letter in letter_count:
            if word_letter_count.get(letter, 0) != letter_count[letter]:
                match = False
                break

        # Check green letters
        if match:
            for i in range(5):
                if feedback[i] == 'green':
                    if word[i] != guess[i]:
                        match = False
                        break

        # Check yellow letters
        if match:
            for i in range(5):
                if feedback[i] == 'yellow':
                    if guess[i] not in word or word[i] == guess[i]:
                        match = False
                        break

                    found_valid = False
                    for j in range(5):
                        if word[j] == guess[i] and j != i and j not in used_indices:
                            used_indices.add(j)
                            found_valid = True
                            break

                    if not found_valid:
                        match = False
                        break

        # Check gray letters
        if match:
            for i in range(5):
                if feedback[i] == 'gray':
                    # Skip gray letters if they were already validated in green/yellow positions
                    if guess[i] in word and guess[i].lower() not in green_yellow_letters:
                        match = False
                        break

        if match:
            filtered_words.append(word)

    return filtered_words

def reset_gray_letters():
    global gray_letters
    gray_letters.clear()
# Wordle Solver Route

@app.route('/wordle', methods=['GET', 'POST'])
def wordle_solver():
    word_list = load_word_list()
    
    if request.method == 'POST':
        data = request.json
        guess = data.get('guess')
        feedback = data.get('feedback')

        # Debugging: Print received data to console
       

        # Filter words based on guess and feedback
        filtered_words = filter_words(word_list, guess, feedback)

        # Provide up to 10 suggestions from filtered words
        suggestions = random.sample(filtered_words, min(10, len(filtered_words)))
        if len(filtered_words)<2:
            reset_gray_letters()
       
        
        return jsonify({'suggestions': suggestions})

    # Render initial HTML page
    return render_template('wordle.html', initial_guesses=[])


def bubble_sort(arr):
    n = len(arr)
    steps = []
    for i in range(n):
        for j in range(0, n-i-1):
            # Highlight the two elements being compared
            steps.append((arr.copy(), [j, j+1]))
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            steps.append((arr.copy(), [j, j+1]))  # After swap or no swap
    return steps

def selection_sort(arr):
    n = len(arr)
    steps = []
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            # Highlight the two elements being compared
            steps.append((arr.copy(), [j, min_idx]))
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        steps.append((arr.copy(), [i, min_idx]))  # After swap
    return steps

def insertion_sort(arr):
    steps = []
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            # Highlight the element being compared
            steps.append((arr.copy(), [j + 1, j]))
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        steps.append((arr.copy(), [j + 1]))  # After insertion
    return steps

def merge_sort(arr):
    steps = []

    def merge(arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m

        L = arr[l:m+1]
        R = arr[m+1:r+1]

        i = j = 0
        k = l

        while i < n1 and j < n2:
            # Highlight the merging elements
            steps.append((arr.copy(), [k, l + i, m + 1 + j]))
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < n1:
            steps.append((arr.copy(), [k, l + i]))  # Highlight remaining elements
            arr[k] = L[i]
            i += 1
            k += 1

        while j < n2:
            steps.append((arr.copy(), [k, m + 1 + j]))  # Highlight remaining elements
            arr[k] = R[j]
            j += 1
            k += 1

    def sort(arr, l, r):
        if l < r:
            m = l + (r - l) // 2
            sort(arr, l, m)
            sort(arr, m + 1, r)
            merge(arr, l, m, r)

    sort(arr, 0, len(arr) - 1)
    steps.append((arr.copy(), []))  # Final state of the array
    return steps

def quick_sort(arr):
    steps = []

    def sort(arr, low, high):
        if low < high:
            # Partition the array
            pivot_index = partition(arr, low, high)
            steps.append((arr.copy(), [pivot_index]))  # Highlight pivot
            # Recursively sort the left and right parts
            sort(arr, low, pivot_index - 1)
            sort(arr, pivot_index + 1, high)

    def partition(arr, low, high):
        pivot = arr[high]  # Choosing the last element as the pivot
        i = low - 1  # Pointer for the smaller element
        for j in range(low, high):
            # Highlight comparison between pivot and element
            steps.append((arr.copy(), [j, high]))
            if arr[j] <= pivot:
                i += 1  # Increment the smaller element index
                arr[i], arr[j] = arr[j], arr[i]
                steps.append((arr.copy(), [i, j]))  # After swap
        # Swap the pivot element with the element at i+1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append((arr.copy(), [i + 1, high]))  # After final pivot swap
        return i + 1

    # Initialize sorting with low as 0 and high as the last index
    sort(arr, 0, len(arr) - 1)
    steps.append((arr.copy(), []))  # Append the final sorted array
    return steps

@app.route('/sort', methods=['GET', 'POST'])
def sort():
    if request.method == 'POST':
        algorithm = request.form.get('algorithm', 'bubble')
        # Regenerate random array each time
        array = [random.randint(1, 100) for _ in range(20)]

        if algorithm == 'bubble':
            steps = bubble_sort(array)
        elif algorithm == 'selection':
            steps = selection_sort(array)
        elif algorithm == 'insertion':
            steps = insertion_sort(array)
        elif algorithm == 'merge':
            steps = merge_sort(array)
        elif algorithm == 'quick':
            steps = quick_sort(array)
        
        return jsonify(steps=steps)
    
    # Initial random array
    array = [random.randint(10, 100) for _ in range(20)]
    return render_template('sort.html', array=array)


if __name__ == '__main__':
    app.run(debug=True)
