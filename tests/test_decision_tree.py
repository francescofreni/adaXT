from adaXT.decision_tree import LeafNode, DecisionNode, DecisionTree
from adaXT.criteria import (
    GiniIndex,
    SquaredError,
    Entropy,
    PartialLinear,
    PartialQuadratic,
    MultiSquaredError,
    PairwiseEuclideanDistance,
)
import numpy as np


def rec_node(node: LeafNode | DecisionNode | None, depth: int) -> None:
    """
    Used to check the depth value associated with nodes

    Parameters
    ----------
    node : LeafNode | DecisionNode | None
        node to recurse on
    depth : int
        expected depth of the node
    """
    if isinstance(node, LeafNode) or isinstance(node, DecisionNode):
        assert node.depth == depth, (
            f"Incorrect depth, expected {depth} got {node.depth}"
        )
        if isinstance(node, DecisionNode):
            rec_node(node.left_child, depth + 1)


def test_gini_single():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=GiniIndex)
    tree.fit(X, Y_cla)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert isinstance(root, LeafNode) or isinstance(root, DecisionNode), (
        f"root is not a node but {type(root)}"
    )
    queue = [root]
    i = 0

    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(
            cur_node, DecisionNode
        ):  # Check threshold and idx of decision node
            assert cur_node.threshold == exp_val[i], (
                f"Expected threshold {exp_val[i]} on node={i}, got {cur_node.threshold} on split_idx {cur_node.split_idx} exp: {spl_idx[i]}"
            )
            assert cur_node.split_idx == spl_idx[i], (
                f"Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}"
            )
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):  # Check that the value is of length 2
            assert len(cur_node.value) == 2, (
                f"Expected 2 mean values, one for each class, but got: {len(cur_node.value)}"
            )

    rec_node(root, 0)


def test_gini_multi():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_multi = np.array([1, 2, 1, 0, 1, 0, 1, 0])
    Y_unique = len(np.unique(Y_multi))
    tree = DecisionTree("Classification", criteria=GiniIndex)
    tree.fit(X, Y_multi)
    root = tree.root
    # DIFFERENT FROM SKLEARN THEIRS IS: [0.25, -0.75, -1.5], both give pure
    # leaf node
    exp_val = [0.25, -0.75, -0.75]
    # DIFFERENT FROM SKLEARN THEIRS IS: [0, 1, 1], both give pure leaf node
    spl_idx = [0, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(root, DecisionNode), (
        f"root is not a node but {type(root)}"
    )
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert cur_node.threshold == exp_val[i], (
                f"Expected threshold {exp_val[i]}, got {cur_node.threshold}"
            )
            assert cur_node.split_idx == spl_idx[i], (
                f"Expected split idx {spl_idx[i]}, got {cur_node.split_idx}"
            )
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert len(cur_node.value) == Y_unique, (
                f"Expected {Y_unique} mean values, one for each class, but got: {len(cur_node.value)}"
            )

    rec_node(root, 0)


def test_regression():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_reg = np.array([2.2, -0.5, 0.5, -0.5, 2, -3, 2.2, -3])
    tree = DecisionTree("Regression", criteria=SquaredError)
    tree.fit(X, Y_reg)
    root = tree.root
    exp_val2 = [0.25, -0.5, 0.5, 0.25, -0.75]
    spl_idx2 = [0, 1, 1, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(root, DecisionNode), (
        f"root is not a node but {type(root)}"
    )
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert cur_node.threshold == exp_val2[i], (
                f"Expected threshold {exp_val2[i]}, got {cur_node.threshold}"
            )
            assert cur_node.split_idx == spl_idx2[i], (
                f"Expected split idx {spl_idx2[i]}, got {cur_node.split_idx}"
            )
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert len(cur_node.value) == 1, (
                f"Expected {1} mean values, but got: {len(cur_node.value)}"
            )
    rec_node(root, 0)


def test_entropy_single():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_cla = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y_cla)
    root = tree.root
    exp_val = [0.25, -0.75, 0]
    spl_idx = [0, 0, 1]
    assert isinstance(root, LeafNode) or isinstance(root, DecisionNode), (
        f"root is not a node but {type(root)}"
    )
    queue = [root]
    i = 0
    # Loop over all the nodes
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(
            cur_node, DecisionNode
        ):  # Check threshold and idx of decision node
            assert cur_node.threshold == exp_val[i], (
                f"Expected threshold {exp_val[i]} on node={i}, got {cur_node.threshold} on split_idx {cur_node.split_idx} exp: {spl_idx[i]}"
            )
            assert cur_node.split_idx == spl_idx[i], (
                f"Expected split idx {spl_idx[i]} on i={i}, got {cur_node.split_idx}"
            )
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):  # Check that the value is of length 2
            assert len(cur_node.value) == 2, (
                f"Expected 2 mean values, one for each class, but got: {len(cur_node.value)}"
            )

    rec_node(root, 0)


def test_entropy_multi():
    X = np.array(
        [
            [1, -1],
            [-0.5, -2],
            [-1, -1],
            [-0.5, -0.5],
            [1, 0],
            [-1, 1],
            [1, 1],
            [-0.5, 2],
        ]
    )
    Y_multi = np.array([1, 2, 1, 0, 1, 0, 1, 0])
    Y_unique = len(np.unique(Y_multi))
    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y_multi)
    root = tree.root
    # DIFFERENT FROM SKLEARN THEIRS IS: [0.25, -0.75, -1.5], both give pure
    # leaf node
    exp_val = [0.25, -0.75, -0.75]
    # DIFFERENT FROM SKLEARN THEIRS IS: [0, 1, 1], both give pure leaf node
    spl_idx = [0, 1, 0]
    assert isinstance(root, LeafNode) or isinstance(root, DecisionNode), (
        f"root is not a node but {type(root)}"
    )
    queue = [root]
    i = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if isinstance(cur_node, DecisionNode):
            assert cur_node.threshold == exp_val[i], (
                f"Expected threshold {exp_val[i]}, got {cur_node.threshold}"
            )
            assert cur_node.split_idx == spl_idx[i], (
                f"Expected split idx {spl_idx[i]}, got {cur_node.split_idx}"
            )
            if cur_node.left_child:
                queue.append(cur_node.left_child)
            if cur_node.right_child:
                queue.append(cur_node.right_child)
            i += 1
        elif isinstance(cur_node, LeafNode):
            assert len(cur_node.value) == Y_unique, (
                f"Expected {Y_unique} mean values, one for each class, but got: {len(cur_node.value)}"
            )

    rec_node(root, 0)


def test_multi_squared():
    np.random.seed(2025)
    N = 500
    Y_M = 4

    # Create clear split on the uneven numbers
    X = np.array([[0, 1] if x % 2 == 0 else [0, 2] for x in range(N)])
    # All the Y values for the even numbers gather around 0.0, and all the
    # values for the uneven numbers gather around 10
    Y = np.array(
        [
            np.random.normal(0.0, 1.0, Y_M)
            if x % 2 == 0
            else np.random.normal(10.0, 1.0, Y_M)
            for x in range(N)
        ]
    )

    tree = DecisionTree(tree_type="MultiRegression")
    tree.fit(X, Y)

    # Should only predict the values of the random normal
    X_pred_1 = np.array([[0, 1] for _ in range(N // 2)])
    X_pred_2 = np.array([[0, 2] for _ in range(N // 2)])
    pred_1 = tree.predict(X_pred_1)
    pred_2 = tree.predict(X_pred_2)
    assert np.all(pred_1 == pred_1[0]), "All elements of pred_1 are not equal"
    assert np.all(pred_2 == pred_2[0]), "All elements of pred_1 are not equal"

    assert abs(np.mean(pred_1) -
               0.0) < 1, "Mean of pred_1 is not approximately 0.0"
    assert abs(np.mean(pred_2) -
               10.0) < 1, "Mean of pred_2 is not approximately 10.0"


def test_Pairwise():
    np.random.seed(2025)
    N = 500
    Y_M = 4

    # Create clear split on the uneven numbers
    X = np.array([[0, 1] if x % 2 == 0 else [0, 2] for x in range(N)])
    # All the Y values for the even numbers gather around 0.0, and all the
    # values for the uneven numbers gather around 10
    Y = np.array(
        [
            np.random.normal(0.0, 1.0, Y_M)
            if x % 2 == 0
            else np.random.normal(10.0, 1.0, Y_M)
            for x in range(N)
        ]
    )

    tree = DecisionTree(
        tree_type="MultiRegression",
        criteria=PairwiseEuclideanDistance)
    tree.fit(X, Y)

    # Should only predict the values of the random normal
    X_pred_1 = np.array([[0, 1] for _ in range(N // 2)])
    X_pred_2 = np.array([[0, 2] for _ in range(N // 2)])
    pred_1 = tree.predict(X_pred_1)
    pred_2 = tree.predict(X_pred_2)
    assert np.all(pred_1 == pred_1[0]), "All elements of pred_1 are not equal"
    assert np.all(pred_2 == pred_2[0]), "All elements of pred_1 are not equal"

    assert abs(np.mean(pred_1) -
               0.0) < 1, "Mean of pred_1 is not approximately 0.0"
    assert abs(np.mean(pred_2) -
               10.0) < 1, "Mean of pred_2 is not approximately 10.0"


def sanity_regression(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y1 = np.random.randint(0, 5, n)
    Y2 = np.random.uniform(0, 5, n)

    tree1 = DecisionTree("Regression", criteria=SquaredError)
    tree2 = DecisionTree("Regression", criteria=SquaredError)
    tree1.fit(X, Y1)
    tree2.fit(X, Y2)
    pred1 = tree1.predict(X)
    pred2 = tree2.predict(X)
    for i in range(n):
        assert abs(Y1[i] - pred1[i]) < 0.00001, (
            f"Square: Expected {Y1[i]} Got {pred1[i]}"
        )
        assert abs(Y2[i] - pred2[i]) < 0.00001, (
            f"Square: Expected {Y2[i]} Got {pred2[i]}"
        )


def sanity_gini(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.randint(0, 5, n)

    tree = DecisionTree("Classification", criteria=GiniIndex)
    tree.fit(X, Y)

    pred = tree.predict(X)
    for i in range(n):
        assert Y[i] == pred[i], f"Gini: Expected {Y[i]} Got {pred[i]}"


def sanity_entropy(n, m):
    X = np.random.uniform(0, 100, (n, m))
    Y = np.random.randint(0, 5, n)

    tree = DecisionTree("Classification", criteria=Entropy)
    tree.fit(X, Y)

    pred = tree.predict(X)
    for i in range(n):
        assert Y[i] == pred[i], f"Entropy: Expected {Y[i]} Got {pred[i]}"


def sanity_partial_linear(n, m):
    X = np.c_[np.linspace(-1, 1, n), np.random.uniform(-1, 1, (n, m))]
    Y = X[:, 0] * (X[:, 0] > 0)
    tree = DecisionTree("Gradient", criteria=PartialLinear, max_depth=1)
    tree.fit(X, Y)
    # Since the response is a piece-wise linear function it can be fit
    # exactly with the PartialLinear criteria, with a single split at 0
    assert (tree.leaf_nodes[0].impurity + tree.leaf_nodes[1].impurity) == 0


def sanity_partial_quadratic(n, m):
    X = np.c_[np.linspace(-1, 1, n), np.random.uniform(-1, 1, (n, m))]
    Y = X[:, 0] ** 2 * (X[:, 0] > 0)
    tree = DecisionTree("Gradient", criteria=PartialQuadratic, max_depth=1)
    tree.fit(X, Y)
    # Since the response is a piece-wise quadratic function it can be fit
    # exactly with the PartialQuadratic criteria, with a single split at 0
    assert (tree.leaf_nodes[0].impurity + tree.leaf_nodes[1].impurity) == 0


def test_sanity():
    n = 10000
    m = 5
    sanity_regression(n, m)
    sanity_gini(n, m)
    sanity_entropy(n, m)
    sanity_partial_linear(n, m)
    sanity_partial_quadratic(n, m)


if __name__ == "__main__":
    # test_gini_single()
    # test_gini_multi()
    # test_entropy_single()
    # test_entropy_multi()
    test_multi_squared()
    print("Done.")
