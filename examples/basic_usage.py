"""
Example usage of trendfilter package.
"""

import numpy as np
import matplotlib.pyplot as plt
from trendfilterpy import TrendFilter, CVTrendFilter


def generate_example_data(n=100, noise_level=0.1):
    """Generate example data for trend filtering."""
    x = np.linspace(0, 1, n)

    # Create a piecewise linear function with some smoothness
    true_signal = np.zeros(n)
    true_signal[0:20] = 0.5 * x[0:20]
    true_signal[20:40] = 0.1 + 0.8 * (x[20:40] - 0.2)
    true_signal[40:60] = 0.26 + 0.2 * (x[40:60] - 0.4)
    true_signal[60:80] = 0.3 - 0.3 * (x[60:80] - 0.6)
    true_signal[80:100] = 0.24 + 0.6 * (x[80:100] - 0.8)

    # Add noise
    y = true_signal + noise_level * np.random.randn(n)

    return x, y, true_signal


def example_basic_usage():
    """Example of basic trend filtering usage."""
    print("=== Basic Trend Filtering Example ===")

    # Generate data
    x, y, true_signal = generate_example_data(n=100, noise_level=0.1)

    # Fit trend filter with single lambda
    tf = TrendFilter(order=1, lambda_reg=0.1)
    tf.fit(y, x)

    print(f"Fitted {len(y)} data points")
    print(f"Lambda used: {tf.lambda_reg}")

    if hasattr(tf, "lambda_"):
        print(f"Actual lambda path length: {len(tf.lambda_)}")
        print(f"Objective values: {getattr(tf, 'objective_', 'Not available')}")

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, y, "o", alpha=0.6, label="Noisy data")
    plt.plot(x, true_signal, "-", linewidth=2, label="True signal")
    plt.plot(x, tf.predict(), "-", linewidth=2, label="Trend filter")
    plt.legend()
    plt.title("Basic Trend Filtering")
    plt.xlabel("x")
    plt.ylabel("y")

    return tf


def example_multiple_lambdas():
    """Example of trend filtering with multiple lambda values."""
    print("\n=== Multiple Lambda Values Example ===")

    # Generate data
    x, y, true_signal = generate_example_data(n=50, noise_level=0.15)

    # Fit trend filter with multiple lambdas
    lambda_values = np.logspace(-3, 1, 20)
    tf = TrendFilter(order=1, lambda_reg=lambda_values)
    tf.fit(y, x)

    print(f"Fitted for {len(lambda_values)} lambda values")

    if hasattr(tf, "degrees_of_freedom_"):
        print(
            f"Degrees of freedom range: {tf.degrees_of_freedom_.min():.1f} - {tf.degrees_of_freedom_.max():.1f}"
        )

    # Plot results for different lambda values
    plt.subplot(1, 2, 2)
    plt.plot(x, y, "o", alpha=0.6, label="Noisy data", markersize=4)
    plt.plot(x, true_signal, "-", linewidth=2, label="True signal", color="black")

    # Plot a few different lambda solutions
    if tf.coef_.ndim > 1:
        for i, idx in enumerate(
            [0, len(lambda_values) // 4, len(lambda_values) // 2, -1]
        ):
            plt.plot(
                x,
                tf.coef_[:, idx],
                "--",
                alpha=0.8,
                label=f"λ = {lambda_values[idx]:.3f}",
            )
    else:
        plt.plot(x, tf.coef_, "--", alpha=0.8, label="Trend filter")

    plt.legend()
    plt.title("Multiple Lambda Values")
    plt.xlabel("x")
    plt.ylabel("y")

    return tf


def example_cross_validation():
    """Example of cross-validated trend filtering."""
    print("\n=== Cross-Validation Example ===")

    # Generate data
    x, y, true_signal = generate_example_data(n=80, noise_level=0.12)

    # Fit with cross-validation
    cv_tf = CVTrendFilter(order=1, cv=5, scoring="mse")
    cv_tf.fit(y, x)

    print(f"Best lambda: {cv_tf.best_lambda_:.4f}")
    print(f"Best CV score: {cv_tf.best_score_:.4f}")
    print(f"Tested {len(cv_tf.lambda_path_)} lambda values")

    # Plot CV results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.semilogx(cv_tf.lambda_path_, cv_tf.cv_scores_, "o-")
    plt.axvline(
        cv_tf.best_lambda_,
        color="red",
        linestyle="--",
        label=f"Best λ = {cv_tf.best_lambda_:.4f}",
    )
    plt.xlabel("Lambda")
    plt.ylabel("CV Score (MSE)")
    plt.title("Cross-Validation Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, y, "o", alpha=0.6, label="Noisy data")
    plt.plot(x, true_signal, "-", linewidth=2, label="True signal")
    plt.plot(x, cv_tf.predict(), "-", linewidth=2, label="CV Trend filter")
    plt.legend()
    plt.title("Cross-Validated Result")
    plt.xlabel("x")
    plt.ylabel("y")

    return cv_tf


def main():
    """Run all examples."""
    print("Trendfilter Examples")
    print("=======================")

    # Check backend availability
    try:
        from trendfilterpy import get_backend_info

        backend_info = get_backend_info()
        print("Backend availability:")
        for backend, available in backend_info.items():
            print(f"  {backend}: {'✓' if available else '✗'}")
        print()
    except ImportError:
        print("Backend info not available\n")

    try:
        # Run examples
        tf1 = example_basic_usage()
        tf2 = example_multiple_lambdas()

        plt.tight_layout()
        plt.show()

        cv_tf = example_cross_validation()
        plt.tight_layout()
        plt.show()

        print("\n=== Summary ===")
        print("Examples completed successfully!")
        print("Check the plots to see trend filtering in action.")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("This might be due to missing dependencies or C++ backend not compiled.")

        # Try simple fallback
        print("\nTrying simple fallback...")
        try:
            x = np.linspace(0, 1, 50)
            y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(50)

            tf = TrendFilter(order=1, lambda_reg=0.1)
            tf.fit(y, x)
            print(f"Fallback successful! Fitted {len(y)} points.")
            print(f"Output shape: {tf.predict().shape}")

        except Exception as e2:
            print(f"Fallback also failed: {e2}")


if __name__ == "__main__":
    main()
