import tempfile

import pytest

from cue.tools.run_script import PythonRunner


@pytest.fixture
def python_runner():
    return PythonRunner(timeout=2)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_simple_script_execution(python_runner):
    script = """
print('Hello, World!')
"""
    result = await python_runner(script)
    assert result.success
    assert result.stdout.strip() == "Hello, World!"
    assert result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_with_allowed_imports(python_runner):
    script = """
import math
import random
print(f'Pi is approximately {math.pi}')
print(f'Random number: {random.randint(1, 100)}')
"""
    result = await python_runner(script)
    assert result.success
    assert "Pi is approximately 3.14" in result.stdout
    assert result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_with_prohibited_imports(python_runner):
    script = """
import os
os.system('echo "This should not work"')
"""
    result = await python_runner(script)
    assert not result.success
    assert "Import of module 'os' is not allowed" in result.stderr
    assert result.exit_code == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_exceeding_timeout(python_runner):
    script = """
import time
while True:
    time.sleep(1)
"""
    result = await python_runner(script)
    assert not result.success
    assert "Script execution timed out" in result.stderr
    assert result.exit_code == 124


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_from_file(python_runner):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write('print("Hello from file!")')
        temp_file.flush()

        result = await python_runner(temp_file.name, is_file=True)

    assert result.success
    assert result.stdout.strip() == "Hello from file!"
    assert result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_with_syntax_error(python_runner):
    script = """
print('Missing closing parenthesis'
"""
    result = await python_runner(script)
    assert not result.success
    assert "Invalid Python syntax" in result.stderr
    assert result.exit_code == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_script_with_runtime_error(python_runner):
    script = """
x = 1 / 0
"""
    result = await python_runner(script)
    assert not result.success
    assert result.stderr != ""
    assert result.exit_code == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_custom_allowed_modules(python_runner):
    # Create runner with only math module allowed
    restricted_runner = PythonRunner(allowed_modules={"math"})

    script = """
import math
import random  # This should fail
print(math.pi)
"""
    result = await restricted_runner(script)
    assert not result.success
    assert "Import of module 'random' is not allowed" in result.stderr
    assert result.exit_code == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_file_size_limit(python_runner):
    # Create a temporary file that exceeds the size limit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        # Write more than MAX_FILE_SIZE bytes
        temp_file.write('print("x" * 2000000)')  # Should exceed 1MB limit
        temp_file.flush()

        result = await python_runner(temp_file.name, is_file=True)

    assert not result.success
    assert "exceeds maximum size" in result.stderr
    assert result.exit_code == 1
