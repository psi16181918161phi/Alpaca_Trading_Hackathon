/*
 * Deck navigation controller.
 *
 * Exposes a small, deterministic API on window so the Playwright recording and
 * PDF scripts drive the deck through the same interface a human keyboard user
 * would -- no scraping of internal state, no timing guesses.
 *
 *   window.XQX.count()      -> total slide count
 *   window.XQX.goto(index)  -> activate slide by zero-based index
 *   window.XQX.next()       -> advance one slide, returns false at the end
 */
(function () {
    "use strict";

    var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
    var current = 0;

    function stamp() {
        slides.forEach(function (slide, index) {
            var number = slide.querySelector(".slide-number");
            if (number) {
                number.textContent = String(index + 1).padStart(2, "0") + " / " + String(slides.length).padStart(2, "0");
            }
        });
    }

    function goto(index) {
        if (index < 0 || index >= slides.length) {
            return false;
        }
        slides.forEach(function (slide) {
            slide.classList.remove("is-active");
        });
        slides[index].classList.add("is-active");
        current = index;
        document.title = "X Quant X - slide " + (index + 1);
        return true;
    }

    function next() {
        return goto(current + 1);
    }

    document.addEventListener("keydown", function (event) {
        if (event.key === "ArrowRight" || event.key === " ") {
            next();
        } else if (event.key === "ArrowLeft") {
            goto(current - 1);
        }
    });

    stamp();
    goto(0);

    window.XQX = {
        count: function () { return slides.length; },
        index: function () { return current; },
        goto: goto,
        next: next
    };
}());
