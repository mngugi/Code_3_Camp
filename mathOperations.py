{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a4b236b8-50d4-43ea-b6e6-a5743c2c4a17",
   "metadata": {},
   "outputs": [],
   "source": [
    "def add(a, b):\n",
    "  return a + b \n",
    "\n",
    "def add(a, b):\n",
    "  return a - b \n",
    "    \n",
    "\n",
    "def add(a, b):\n",
    "  return a * b \n",
    "\n",
    "def add(a, b):\n",
    "  return a / b \n",
    "        \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "44567798-f7c7-4e6d-977b-82724772535b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.core.magic import register_line_magic\n",
    "\n",
    "@register_line_magic\n",
    "def math_operations(line):\n",
    "    \"\"\"\n",
    "    Magic command to perform basic mathematical operations.\n",
    "\n",
    "    Usage:\n",
    "    %math_operations operation operand1 operand2\n",
    "\n",
    "    Supported operations: add, subtract, multiply, divide\n",
    "\n",
    "    Example:\n",
    "    %math_operations add 6 2\n",
    "    \"\"\"\n",
    "    # Parse the input\n",
    "    parts = line.split()\n",
    "    if len(parts) != 3:\n",
    "        print(\"Usage: %math_operations operation operand1 operand2\")\n",
    "        return\n",
    "\n",
    "    operation, operand1, operand2 = parts\n",
    "\n",
    "    try:\n",
    "        operand1 = float(operand1)\n",
    "        operand2 = float(operand2)\n",
    "    except ValueError:\n",
    "        print(\"Error: Operands must be numeric.\")\n",
    "        return\n",
    "\n",
    "    # Perform the operation\n",
    "    result = None\n",
    "    if operation == 'add':\n",
    "        result = operand1 + operand2\n",
    "    elif operation == 'subtract':\n",
    "        result = operand1 - operand2\n",
    "    elif operation == 'multiply':\n",
    "        result = operand1 * operand2\n",
    "    elif operation == 'divide':\n",
    "        if operand2 == 0:\n",
    "            print(\"Error: Cannot divide by zero.\")\n",
    "            return\n",
    "        result = operand1 / operand2\n",
    "    else:\n",
    "        print(\"Error: Unsupported operation.\")\n",
    "        return\n",
    "\n",
    "    print(\"Result:\", result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "883162a8-311a-489c-b5d0-1611aafe90ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Result: 8.0\n",
      "Result: 4.0\n",
      "Result: 12.0\n",
      "Result: 3.0\n"
     ]
    }
   ],
   "source": [
    "# Use the magic command\n",
    "%math_operations add 6 2\n",
    "%math_operations subtract 6 2\n",
    "%math_operations multiply 6 2\n",
    "%math_operations divide 6 2\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
