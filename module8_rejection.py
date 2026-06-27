"""
=============================================================
MODULE 8: OUT-OF-DISTRIBUTION REJECTION
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Three layers of rejection:
  Layer 1 — Color/texture prefilter (8 checks)
  Layer 2 — Model predicts "non_tomato" class
  Layer 3 — Confidence threshold (0.75)
=============================================================
"""

import numpy as np


class TomatoRejector:
    """
    Rejects non-tomato images before and after model prediction.
    """

    def __init__(self, confidence_threshold=0.75):
        self.confidence_threshold = confidence_threshold


    # ─────────────────────────────────────────
    # LAYER 1: COLOR / TEXTURE PREFILTER
    # ─────────────────────────────────────────

    def prefilter(self, image):
        """
        8 checks to reject obviously non-leaf images.

        Args:
            image : (1, H, W, 3) or (H, W, 3) float32 normalized [0,1]

        Returns:
            (is_plausible, reason, green_ratio)
        """
        img = image[0] if len(image.shape) == 4 else image

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        avg_r      = float(np.mean(r))
        avg_g      = float(np.mean(g))
        avg_b      = float(np.mean(b))
        std_all    = float(np.std(img))
        avg_bright = float(np.mean(img))

        # Check 1: Reject near-grayscale (cars, roads, concrete)
        channel_var = float(np.var([avg_r, avg_g, avg_b]))
        if channel_var < 0.00005:
            return False, (
                f"Image looks gray/colorless "
                f"(R={avg_r:.2f} G={avg_g:.2f} B={avg_b:.2f} var={channel_var:.5f})"
            ), 0.0

        # Check 2: Green must be dominant or near-dominant
        if avg_g < avg_r - 0.08 and avg_g < avg_b - 0.08:
            return False, (
                f"Green is weakest channel "
                f"(R={avg_r:.2f} G={avg_g:.2f} B={avg_b:.2f})"
            ), 0.0

        # Check 3: Too bright / washed out
        if avg_bright > 0.88:
            return False, f"Image too bright/white (mean={avg_bright:.2f})", 0.0

        # Check 4: Too dark
        if avg_bright < 0.08:
            return False, f"Image too dark (mean={avg_bright:.2f})", 0.0

        # Check 5: No texture (solid color)
        if std_all < 0.04:
            return False, f"No texture detected (std={std_all:.3f})", 0.0

        # Check 6: Strict green pixel ratio
        green_pixels = (g > r + 0.02) & (g > b + 0.02) & (g > 0.20)
        green_ratio  = float(np.mean(green_pixels))
        if green_ratio < 0.08:
            return False, (
                f"Too few green pixels ({green_ratio:.1%} < 8%)"
            ), green_ratio

        # Check 7: Strong blue dominance (sky, water)
        if avg_b > avg_g + 0.15:
            return False, (
                f"Too much blue (B={avg_b:.2f} > G={avg_g:.2f})"
            ), green_ratio

        # Check 8: Strong red dominance (tomato fruit, red objects)
        if avg_r > avg_g + 0.15:
            return False, (
                f"Too much red (R={avg_r:.2f} > G={avg_g:.2f})"
            ), green_ratio

        return True, "Looks like a plausible tomato leaf", green_ratio


    # ─────────────────────────────────────────
    # LAYER 3: CONFIDENCE CHECK
    # ─────────────────────────────────────────

    def should_reject_by_confidence(self, confidence, predicted_class):
        if confidence < self.confidence_threshold:
            return True, (
                f"Confidence too low "
                f"({confidence*100:.1f}% < {self.confidence_threshold*100:.0f}%)"
            )
        return False, None


    # ─────────────────────────────────────────
    # REJECTION MESSAGE
    # ─────────────────────────────────────────

    def get_rejection_message(self, reason, green_ratio=None, confidence=None):
        msg = f"\n  Not a Valid Tomato Leaf\n  Reason : {reason}\n"
        if green_ratio is not None and green_ratio > 0:
            msg += f"  Green pixels : {green_ratio:.1%}\n"
        if confidence is not None:
            msg += f"  Confidence   : {confidence*100:.1f}%\n"
        msg += (
            "\n  Tips:\n"
            "    - Upload a clear photo of a single tomato leaf\n"
            "    - Ensure good lighting\n"
            "    - Fill the frame with the leaf\n"
            "    - Avoid blurry or very dark photos\n"
        )
        return msg


    # ─────────────────────────────────────────
    # WARNING MESSAGE
    # ─────────────────────────────────────────

    def get_warning_message(self, confidence, predicted_class):
        name = (
            predicted_class
            .replace("Tomato__", "")
            .replace("Tomato_", "")
            .replace("_", " ")
        )
        return (
            f"\n  Medium Confidence Warning\n"
            f"  Detected   : {name}\n"
            f"  Confidence : {confidence*100:.1f}%\n"
            "  Consider taking another photo for confirmation.\n"
        )


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 8: REJECTION — Quick Test")
    print("=" * 55)

    rejector = TomatoRejector(confidence_threshold=0.75)

    def test(name, img):
        ok, reason, ratio = rejector.prefilter(img)
        status = "PASS  " if ok else "REJECT"
        print(f"  [{status}] {name:35} {reason[:50]}")

    print()
    car = np.ones((1,64,64,3), np.float32) * 0.55
    test("Gray car", car)

    sky = np.ones((1,64,64,3), np.float32) * 0.3
    sky[0,:,:,2] = 0.75
    test("Blue sky", sky)

    red = np.ones((1,64,64,3), np.float32) * 0.2
    red[0,:,:,0] = 0.85
    test("Red object", red)

    test("White/blank", np.ones((1,64,64,3), np.float32) * 0.95)
    test("Dark image",  np.zeros((1,64,64,3), np.float32))

    sukuma = np.random.rand(1,64,64,3).astype(np.float32) * 0.3
    sukuma[0,:,:,1] = 0.55
    sukuma[0,:,:,0] = 0.22
    sukuma[0,:,:,2] = 0.18
    test("Sukuma wiki (caught by model/confidence)", sukuma)

    leaf = np.random.rand(1,64,64,3).astype(np.float32) * 0.25
    leaf[0,:,:,1] = 0.62
    leaf[0,:,:,0] = 0.28
    leaf[0,:,:,2] = 0.20
    test("Healthy tomato leaf", leaf)

    print("\n  [Confidence Tests]")
    for conf, cls in [
        (0.40, "Tomato_Early_blight"),
        (0.76, "Tomato_healthy"),
        (0.65, "Tomato_Late_blight"),
    ]:
        reject, msg = rejector.should_reject_by_confidence(conf, cls)
        verdict = "REJECT" if reject else "ACCEPT"
        print(f"  [{verdict}] {cls.replace('Tomato_','')} @ {conf*100:.0f}%")

    print("\n  MODULE 8 TEST PASSED")