from evals.hello_world.hello_world import TaskFamily


def task_family():
    return TaskFamily()


def test_hello_world(task_family: TaskFamily):
    score = task_family.score(task_family.get_tasks()["1"], "hello world!")
    assert score == 1.0
