#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include <dice/template-library/bitset.hpp>
#include <dice/template-library/sandbox.hpp>

#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <new>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Reproducers for the review threads on PR #208. Each test names its thread; links are
// https://github.com/dice-group/dice-template-library/pull/208#discussion_r<id>.
// They encode the behaviour the review asks for, so a test goes green as its finding is fixed.
// All of them are green as of this commit.
//
// r3674844666 formatter needs a test          -> "formatter"
// r3892270977 remove bitset(size_t)           -> "bitset(size_t) does not mean what std::bitset means"
// r3892361831 default ctor leaves garbage     -> "default-initialized fixed bitset"
// r3892375761 logical_size returns max_bits    -> "logical size ignores bits when it sits below max_bits"
// r3892379162 ctor allows bits > max_bits     -> "initializer list can exceed max_bits"
// r3892392716 padding counted by any/none_set -> "any_set/none_set see padding bits"
// r3892394071 padding counted by operator==   -> "operator== compares padding bits"
// r3892398232 shrink_to_fit grows size        -> "shrink_to_fit grows the logical size"
// r3892403144 proxy op= missing !is_const     -> "const iterator is writable"
// r3892404711 proxy op= throws under noexcept -> "proxy assignment throws through noexcept"
// r3892406755 proxy op bool throws, noexcept  -> "proxy operator bool throws through noexcept"
// r3892413481 flip/reset inconsistent w/ set  -> "set/reset/flip disagree about growing"
// r3892417474 remove reversed operator-       -> "n + it is required, n - it is not"
// r3892421436 operator-= clamps to zero       -> "operator-= reports stepping past begin() ..."
// r3892435572 do the doc examples work?       -> "class documentation examples"
// r3892440968 format {:b} terminates          -> "format {:b} does not terminate the process"
//
// Found while writing the above, not raised in the review:
//   "formatter prints the most significant segment first ..." - the formatter was MSB-first inside
//     a segment but put segment 0 first, so its digits matched neither boost/std::bitset nor the
//     bitset's own rbegin()
//   "out-of-range indices are rejected by every mutator" - set/test/set_positions threw where
//     reset/flip/reset_positions silently ignored, for the same index on the same object
//   "right shift no longer pulls padding into the logical range" - the ctor rounded the logical size
//     up to a whole segment, and >>= then shifted that padding down into real bits;
//     "initializer list sizes the bitset to its exact bit width" states the contract replacing it
//   "const iterator is writable" also covers std::ranges::output_range<bitset const, bool>
//   not testable: a fixed bitset whose bits and max_bits round to *different* segment counts
//     (e.g. bitset<10, 100, uint8_t>) fails to compile with "implicit instantiation of undefined
//     template flex_array_inner<unsigned char, 2, 13>" instead of a diagnosable static_assert
//   "formatter" also covers hex zero-padding for wide segments and a repeated 'b' in the spec
//
// Found by the differential tests at the bottom of this file, both about the same confusion between
// the logical size and the storage size:
//   "left shift keeps the bits it moves past the old logical size" - operator<<= grew the logical
//     size but masked the top segment with the *old* leftover width first, clearing what it had
//     just shifted in
//   "operations follow the logical size, not the storage size" - full_segments_or() put the boundary
//     at the last storage segment rather than at the logical size, and operator>>=, resize() and
//     shrink_to_fit() all leave those a whole segment apart. It also covers combining two bitsets
//     of equal size whose storages differ, which read off the end of the shorter one, and the
//     logical size overshooting max_bits in operator<<=
//
// No change needed:
//   r3758384352 (drop the const on the proxy's operator=) - `const reference& operator=(bool) const
//     noexcept` is verbatim the signature the working draft gives std::bitset::reference in
//     [template.bitset.general], and it is load-bearing: std::indirectly_writable assigns through a
//     const rvalue of the reference type, so without the qualifier the proxy is not writable at all
//     and std::ranges::fill/copy over a bitset stop compiling. libstdc++ has not implemented that
//     overload for std::bitset yet, but it did add it for vector<bool>'s proxy in C++23.
//
// Not expressible as a test (style/design, or answered in-thread): r3619903969, r3655021543,
// r3655025644, r3655043492, r3655046897, r3655050529, r3655051579, r3655073150, r3674765934,
// r3892333780, r3892345774 (whether segments_transform_with()'s own size check is reachable is a
// static property; that its three callers throw first is already covered by tests_bitset.cpp's
// "bitwise combination with mismatched sizes throws, leaving the receiver untouched"),
// r3892348228, r3892353849, r3892431856 (missing includes - only a compile-time
// concern, currently satisfied transitively), r3892437370 (unused: `segments_all_of` and the
// const overload of `full_segment()` have no callers; `parse()`'s second `if` is dead - see the
// duplicate-flag check in "formatter").

// the '0'/'1' characters of a "{:b}"-formatted bitset, in output order, framing stripped
std::string binary_digits(std::string_view const formatted) {
	std::string out;
	for (char const c : formatted) {
		if (c == '0' || c == '1') {
			out += c;
		}
	}
	return out;
}

// the hex digits of a "{}"-formatted bitset, in output order, "0x" prefixes and framing stripped
std::string hex_digits(std::string_view const formatted) {
	std::string out;
	bool in_number = false;
	for (char const c : formatted) {
		if (c == 'x') {
			in_number = true;
		} else if (c == ']') {
			in_number = false;
		} else if (in_number) {
			out += c;
		}
	}
	return out;
}

// requires-expressions need a real template parameter so an ill-formed expression fails by
// substitution instead of becoming a hard error.
template<typename I>
concept deref_is_assignable = requires(I it, bool b) { *it = b; };

template<typename I>
concept has_reversed_difference = requires(I it, std::iter_difference_t<I> n) { n - it; };

TEST_SUITE("bitset review findings") {
	using namespace dice::template_library;

	using dyn8 = bitset<dynamic_extent, dynamic_extent, uint8_t>;
	using dyn64 = bitset<dynamic_extent, dynamic_extent>;
	using capped8 = bitset<dynamic_extent, 10, uint8_t>;  // 10 logical bits, 2 uint8_t segments
	using fixed8 = bitset<10, 10, uint8_t>;               // same, but fully static

	TEST_CASE("default-initialized fixed bitset") {
		// r3892361831: the fully static storage has no initializer, so `fixed8 b;` adopts whatever
		// was in that memory. Dirty a buffer and default-initialize into it to make that visible.
		alignas(fixed8) std::array<std::byte, sizeof(fixed8)> buf{};
		auto *dirty = reinterpret_cast<volatile std::byte *>(buf.data());
		for (size_t i = 0; i < buf.size(); ++i) {
			dirty[i] = std::byte{0xff};  // volatile, so the fill cannot be optimized away
		}

		auto const *b = new (static_cast<void *>(buf.data())) fixed8;  // default-init, not `fixed8{}`
		CHECK_EQ(b->count(), 0);
		CHECK_FALSE(b->any_set());
		CHECK(b->none_set());

		// value-initialization is only saved by zero-init happening before the defaulted ctor runs
		fixed8 const value_initialized{};
		CHECK_EQ(value_initialized.count(), 0);
	}

	TEST_CASE("initializer list can exceed max_bits") {
		// r3892379162: bits_ is set from the list width without consulting max_bits, so the bitset
		// claims 16 logical bits while refusing every index >= 10.
		CHECK_THROWS_AS((capped8{0xff, 0xff}), std::length_error);
	}

	TEST_CASE("shrink_to_fit grows the logical size") {
		// r3892398232: bits_ is recomputed as a whole number of segments, which rounds up.
		SUBCASE("uncapped") {
			dyn64 b{};
			b.set(3);
			REQUIRE_EQ(b.size_in_bits(), 4);

			b.shrink_to_fit();
			CHECK_EQ(b.size_in_bits(), 4);
			CHECK_EQ(b.count(), 1);

			dyn64 unshrunk{};
			unshrunk.set(3);
			CHECK(unshrunk == b);  // and the round-up breaks equality
		}

		SUBCASE("capped") {
			bitset<dynamic_extent, 100> b{};
			b.set(3);
			b.shrink_to_fit();
			CHECK_EQ(b.size_in_bits(), 4);
		}
	}

	TEST_CASE("set/reset/flip disagree about growing") {
		// r3892413481: set() grows the logical size, reset() and flip() bail out instead. Whichever
		// convention wins, the three have to agree.
		dyn64 via_set{};
		via_set.set(5, false);
		dyn64 via_reset{};
		via_reset.reset(5);
		dyn64 via_flip{};
		via_flip.flip(5);

		REQUIRE_EQ(via_set.size_in_bits(), 6);
		CHECK_EQ(via_flip.size_in_bits(), via_set.size_in_bits());
		CHECK(via_flip.test(5));  // flip() dropped the bit entirely
	}

	TEST_CASE("out-of-range indices are rejected by every mutator") {
		// not from the review: index 20 used to be a hard error for half the mutators on a capped
		// bitset and a silent no-op for the other half. All six agree on set()'s contract now.
		capped8 b{};
		b.set(9);
		REQUIRE_EQ(b.size_in_bits(), 10);  // == max_bits

		std::vector<size_t> const out_of_range{20};
		CHECK_THROWS_AS(b.set(20), std::out_of_range);
		CHECK_THROWS_AS((void)b.test(20), std::out_of_range);
		CHECK_THROWS_AS(b.set_positions(out_of_range), std::out_of_range);

		CHECK_THROWS_AS(b.reset(20), std::out_of_range);
		CHECK_THROWS_AS(b.flip(20), std::out_of_range);
		CHECK_THROWS_AS(b.reset_positions(out_of_range), std::out_of_range);
	}

	TEST_CASE("right shift no longer pulls padding into the logical range") {
		// the initializer-list ctor used to round the logical size up to a whole segment, so >>=
		// shifted those padding bits down into the logical range (16 logical bits for a 10-bit
		// value, of which 6 were padding). The ctor now sizes the bitset to the exact bit width of
		// the list, so there is no padding left to pull in and >>= only moves real bits.
		SUBCASE("capped") {
			capped8 b{0x00, 0x03};
			REQUIRE_EQ(b.size_in_bits(), 10);
			REQUIRE_EQ(b.count(), 2);

			b >>= 4;
			CHECK_EQ(b.size_in_bits(), 6);
			CHECK_EQ(b.count(), 2);
			CHECK(b.test(4));
			CHECK(b.test(5));
		}

		SUBCASE("uncapped") {
			dyn8 b{0x00, 0x03};
			REQUIRE_EQ(b.size_in_bits(), 10);
			REQUIRE_EQ(b.count(), 2);

			b >>= 4;
			CHECK_EQ(b.size_in_bits(), 6);
			CHECK_EQ(b.count(), 2);
			CHECK(b.test(4));
			CHECK(b.test(5));
		}
	}

	TEST_CASE("left shift keeps the bits it moves past the old logical size") {
		// not from the review, found by the differential tests below: operator<<= grew the logical
		// size to bits_ + shift, but masked the top segment with the *old* leftover width first,
		// which cleared every bit that had just landed in the newly exposed range.
		SUBCASE("a single bit survives the shift") {
			dyn8 b{};
			b.set(9);
			REQUIRE_EQ(b.size_in_bits(), 10);
			REQUIRE_EQ(b.capacity_in_bits(), 16);

			b <<= 1;
			CHECK_EQ(b.size_in_bits(), 11);
			CHECK_EQ(b.count(), 1);  // the bit moved from 9 to 10, it did not disappear
			CHECK(b.test(10));
		}

		SUBCASE("only the bits shifted past the capacity are lost") {
			dyn8 b{0xff, 0x03};  // 10 logical bits, all set
			REQUIRE_EQ(b.count(), 10);

			b <<= 4;
			CHECK_EQ(b.size_in_bits(), 14);
			CHECK_EQ(b.count(), 10);  // bits 4..13, all still inside the 16-bit capacity
		}
	}

	TEST_CASE("operations follow the logical size, not the storage size") {
		// not from the review, found by the differential tests below: full_segments_or() used to put
		// the boundary at the last *storage* segment. operator>>=, resize() and shrink_to_fit() all
		// leave the storage holding more segments than the logical size spans, so that assumption
		// was wrong and everything built on it worked on the wrong segment.
		SUBCASE("the segments above the logical size hold nothing") {
			dyn8 b{};
			b.set(9);
			b >>= 4;  // 6 logical bits, still 2 segments of storage
			REQUIRE_EQ(b.size_in_bits(), 6);
			REQUIRE_EQ(b.capacity_in_bits(), 16);

			b.set_all();
			CHECK_EQ(b.count(), 6);  // only the 6 logical bits exist
			CHECK(b.all_set());
			CHECK_EQ(b.countr_one(), 6);
			CHECK_EQ(b.countl_one(), 6);
			CHECK_EQ(std::ranges::distance(b.positions()), 6);

			auto const flipped = ~b;
			CHECK(flipped.none_set());  // ~ must not hand back the segment above the logical end
			CHECK_EQ(flipped.size_in_bits(), 6);
		}

		SUBCASE("equal-sized bitsets combine even when their storage differs") {
			dyn8 wide{};
			wide.set(9);
			wide >>= 4;         // 6 logical bits across 2 segments
			dyn8 const narrow{0x21};  // 6 logical bits in 1 segment
			REQUIRE_EQ(wide.size_in_bits(), narrow.size_in_bits());
			REQUIRE(wide.capacity_in_bits() > narrow.capacity_in_bits());

			// the shorter storage has no second segment to read
			CHECK_EQ((wide | narrow).count(), 2);
			CHECK_EQ((wide & narrow).count(), 1);
			CHECK_EQ((wide ^ narrow).count(), 1);
		}

		SUBCASE("equality is about the logical bits, whatever storage holds them") {
			dyn8 wide{};
			wide.set(9);
			wide >>= 4;  // bit 5 of 6, in 2 segments
			dyn8 const narrow{0x20};

			CHECK(wide == narrow);
			CHECK(narrow == wide);
			CHECK_FALSE(narrow == dyn8{0xa0});  // same bit pattern, 8 logical bits
		}

		SUBCASE("resize() down drops the bits above the new size") {
			dyn8 b{0xff};
			b.resize(3);
			CHECK_EQ(b.size_in_bits(), 3);
			CHECK_EQ(b.count(), 3);

			b.resize(8);  // the bits it dropped do not come back
			CHECK_EQ(b.count(), 3);
		}

		SUBCASE("shrink_to_fit() never leaves the logical size above the storage") {
			dyn8 b{};
			b.set(9);
			b.reset(9);  // logical size stays at 10, but segment 1 is empty again
			b.set(0);

			b.shrink_to_fit();
			CHECK_EQ(b.capacity_in_bits(), 8);
			CHECK_EQ(b.size_in_bits(), 8);  // clamped - it cannot claim bits the storage dropped
			CHECK_EQ(b.count(), 1);
		}

		SUBCASE("<<= grows the logical size within the cap, not within the capacity") {
			capped8 b{};
			b.set(9);  // 10 logical bits == max_bits, but 2 segments == 16 bits of storage
			REQUIRE_EQ(b.capacity_in_bits(), 16);

			b <<= 3;
			CHECK_EQ(b.size_in_bits(), 10);  // max_bits, not the 16 the storage would allow
			CHECK_EQ(b.count(), 0);          // bit 9 moved to 12, which is past the cap
		}
	}

	TEST_CASE("initializer list sizes the bitset to its exact bit width") {
		// the counterpart to the two cases above: what the ctor now does, stated directly. Trailing
		// all-zero segments are dropped and the top segment contributes only its significant bits,
		// so the logical size is the width of the value while the storage stays whole segments.
		SUBCASE("logical size is the bit width of the value, not a whole segment") {
			dyn8 const b{0x00, 0x03};
			CHECK_EQ(b.size_in_bits(), 10);      // 8 + 2 significant bits
			CHECK_EQ(b.capacity_in_bits(), 16);  // storage still rounds up to whole segments
			CHECK_EQ(b.count(), 2);
			CHECK(b.test(8));
			CHECK(b.test(9));
			CHECK_FALSE(b.test(10));  // past the logical size - no padding bit to observe
		}

		SUBCASE("a single significant bit gives a single logical bit") {
			dyn8 const b{0b00000001};
			CHECK_EQ(b.size_in_bits(), 1);
			CHECK_EQ(b.capacity_in_bits(), 8);
			CHECK_EQ(b.count(), 1);
		}

		SUBCASE("trailing all-zero segments are dropped") {
			dyn8 const b{0xff, 0x00, 0x00};
			CHECK_EQ(b.size_in_bits(), 8);
			CHECK_EQ(b.capacity_in_bits(), 8);
			CHECK_EQ(b.count(), 8);
		}

		SUBCASE("an all-zero list is the empty bitset") {
			dyn8 const b{0x00, 0x00};
			CHECK_EQ(b.size_in_bits(), 0);
			CHECK_EQ(b.capacity_in_bits(), 0);
			CHECK(b.none_set());
		}

		SUBCASE("a top segment with its MSB set stays segment-aligned") {
			dyn8 const b{0x12, 0xb4};
			CHECK_EQ(b.size_in_bits(), 16);
			CHECK_EQ(b.count(), 6);  // 0x12 has 2 bits set, 0xb4 has 4
		}

		SUBCASE("the exact width is what max_bits is checked against") {
			// 0x03 in the top segment is 10 bits wide, which still fits a 10-bit cap ...
			CHECK_NOTHROW((capped8{0x00, 0x03}));
			CHECK_NOTHROW((fixed8{0x00, 0x03}));
			// ... while one more significant bit does not
			CHECK_THROWS_AS((capped8{0x00, 0x07}), std::length_error);
			CHECK_THROWS_AS((fixed8{0x00, 0x07}), std::length_error);
		}

		SUBCASE("a fixed bitset keeps its declared size regardless of the list") {
			fixed8 const b{0x01};
			CHECK_EQ(b.size_in_bits(), 10);  // bits == max_bits == 10, not the list's 1
			CHECK_EQ(b.count(), 1);
		}
	}

	TEST_CASE("n + it is required, n - it is not") {
		// r3892417474: the mirrored operator- was removed because n - it is not subtraction. The
		// mirrored operator+ has to stay - std::random_access_iterator asks for n + i by name, and
		// bit_iterator advertises random_access_iterator_tag.
		CHECK_FALSE(has_reversed_difference<dyn64::bit_iterator>);
		CHECK_FALSE(has_reversed_difference<dyn64::const_bit_iterator>);

		CHECK(std::random_access_iterator<dyn64::bit_iterator>);
		CHECK(std::random_access_iterator<dyn64::const_bit_iterator>);

		dyn8 b{0x12, 0x34};
		auto const it = b.begin() + 2;
		CHECK((5 + it) == (it + 5));
		CHECK_EQ((5 + it).get(), (it + 5).get());
	}

	TEST_CASE("const iterator is writable") {
		// r3892403144: reference::operator=(bool) is missing `&& !is_const`, so a const_iterator's
		// proxy passes every writability check (it only fails once instantiated).
		CHECK(deref_is_assignable<dyn64::bit_iterator>);
		CHECK_FALSE(deref_is_assignable<dyn64::const_bit_iterator>);

		CHECK_FALSE(std::indirectly_writable<dyn64::const_bit_iterator, bool>);
		CHECK_FALSE((std::output_iterator<dyn64::const_bit_iterator, bool>) );

		// so a const bitset passes as a writable range: std::ranges::fill(b, true) on a const
		// bitset satisfies every concept and only fails once it is instantiated
		CHECK_FALSE((std::ranges::output_range<dyn64 const, bool>) );
		CHECK((std::ranges::output_range<dyn64, bool>) );
	}

	TEST_CASE("operator-= reports stepping past begin() instead of clamping") {
		// r3892421436: stepping back past index 0 used to land on 0, which silently broke the
		// (i -= n) += n == i and (i - n) - i == -n axioms. There is no iterator before begin() to
		// return, so the step is now rejected.
		dyn64 b{};
		b.set(70);
		auto const it = b.begin() + 3;

		CHECK_THROWS_AS((void) (it - 5), std::out_of_range);

		auto mut = it;
		CHECK_THROWS_AS(mut -= 5, std::out_of_range);
		CHECK(mut == it);  // and the rejected step leaves the iterator where it was

		// += with a negative delta routes through operator-= and is rejected the same way
		CHECK_THROWS_AS(mut += -5, std::out_of_range);
		CHECK(mut == it);

		// within range the axioms hold
		auto const before = it - 3;
		CHECK(before == b.begin());
		CHECK_EQ(before - it, -3);
		CHECK((before + 3) == it);
		CHECK_EQ((*(before + 3)).ix(), 3);

		// stepping exactly to begin() is in range; one further is not
		CHECK_NOTHROW((void) (b.begin() - 0));
		CHECK_THROWS_AS((void) (b.begin() - 1), std::out_of_range);
	}

	TEST_CASE("class documentation examples") {
		// r3892435572: the doc block's static and fixed examples use 3 segments and index 42, both
		// of which exceed a 10-bit bitset.
		SUBCASE("dynamic") {
			dyn8 b{0x12, 0x13, 0x14};
			CHECK_NOTHROW(b.set(4000uz));
		}

		SUBCASE("static") {
			CHECK_THROWS_AS((capped8{0x12, 0x13, 0x14}), std::length_error);

			capped8 b{0x12, 0x02};  // the widest list the type accepts
			CHECK_NOTHROW(b.set(6uz));
		}

		SUBCASE("fixed") {
			CHECK_THROWS_AS((fixed8{0x12, 0x13, 0x14}), std::length_error);

			fixed8 b{0x12, 0x02};
			CHECK_NOTHROW(b.set(6uz));
		}
	}

	TEST_CASE("formatter") {
		// r3674844666: tests_bitset.cpp covers the formatter for segment-aligned bitsets and valid
		// and invalid specs by now. These are the gaps that coverage still leaves.
		SUBCASE("printing covers exactly the segments the logical bits live in") {
			// root cause of r3892440968: the loop used to walk bit positions through the proxy and
			// ran past size_in_bits(). It now walks segments and reads each one directly, so a
			// logical size that ends mid-segment prints that segment in full and stops there -
			// two segments here, not one and not three.
			dyn8 b{};
			b.set(9);
			REQUIRE_EQ(b.size_in_bits(), 10);
			REQUIRE_EQ(b.capacity_in_bits(), 16);

			CHECK_EQ(std::format("{:b}", b), "[\n[00000010]\n[00000000]\n]\n");
			CHECK_EQ(binary_digits(std::format("{:b}", b)).size(), 16);
		}

		SUBCASE("hex zero padding is two digits short for wide segments") {
			// the width passed to {:#0{}x} is sizeof(T) * 2 and does not account for the "0x"
			// prefix. For uint8_t the width never binds (already characterised in tests_bitset.cpp
			// as "hex mode drops a fully-zero segment's leading zero"), but for uint64_t it does.
			dyn64 const b{0x12};
			CHECK_EQ(std::format("{}", b), "[\n[0x0000000000000012]\n]\n");
		}

		SUBCASE("a repeated b is rejected like any other bad spec") {
			// parse()'s second `if (*it == 'b')` is unreachable, so duplicates slip through
			dyn8 b{0x12};
			std::string s;
			CHECK_THROWS_AS(s = std::vformat("{:bb}", std::make_format_args(b)), std::format_error);
		}
	}

	TEST_CASE("formatter prints the most significant segment first, MSB-first within a segment") {
		// not from the review: the formatter used to print MSB-first *within* a segment but put
		// segment 0 *first*, so its digits read as neither of the two self-consistent orders. It
		// now agrees with boost, std::bitset and the bitset's own rbegin(), and each segment is
		// rendered exactly as it sits in memory rather than truncated to the logical size.
		SUBCASE("two segments") {
			dyn8 const b{0x12, 0xb4};          // bits 0..7 = 0x12, bits 8..15 = 0xb4, value 0xb412
			REQUIRE_EQ(b.size_in_bits(), 16);  // the top segment's MSB is set, so nothing is trimmed

			boost::dynamic_bitset<uint8_t> const ref(16, 0xb412ul);
			std::bitset<16> const std_ref{0xb412ull};
			for (size_t i = 0; i < 16; ++i) {
				REQUIRE_EQ(b.test(i), ref.test(i));
				REQUIRE_EQ(b.test(i), std_ref.test(i));
			}

			std::string want;
			boost::to_string(ref, want);
			REQUIRE_EQ(want, std_ref.to_string());

			// reverse iteration over the same bitset produces boost's order
			std::string via_rbegin;
			for (dyn8::const_reverse_iterator it = b.rbegin(); it != b.rend(); ++it) {
				via_rbegin += static_cast<bool>(*it) ? '1' : '0';
			}
			REQUIRE_EQ(via_rbegin, want);

			CHECK_EQ(binary_digits(std::format("{:b}", b)), want);
			CHECK_EQ(hex_digits(std::format("{}", b)), "b412");
		}

		SUBCASE("a logical size ending mid-segment still prints whole segments") {
			// same two segments, but the top one's high bits are zero, so the exact-width ctor
			// leaves the logical size at 14. The formatter shows the segment as it is in memory,
			// which is boost's 16-bit string; rbegin() covers the 14 logical bits, i.e. the same
			// digits minus the leading two.
			dyn8 const b{0x12, 0x34};
			REQUIRE_EQ(b.size_in_bits(), 14);

			boost::dynamic_bitset<uint8_t> const ref(16, 0x3412ul);
			std::string want;
			boost::to_string(ref, want);

			CHECK_EQ(binary_digits(std::format("{:b}", b)), want);
			CHECK_EQ(hex_digits(std::format("{}", b)), "3412");

			std::string via_rbegin;
			for (dyn8::const_reverse_iterator it = b.rbegin(); it != b.rend(); ++it) {
				via_rbegin += static_cast<bool>(*it) ? '1' : '0';
			}
			CHECK_EQ(via_rbegin, want.substr(2));
		}

		SUBCASE("three segments pin down the segment order") {
			dyn8 const b{0x01, 0x02, 0x04};  // bits 0, 9 and 18

			// built over the whole storage, since that is what the formatter renders
			std::string want;
			for (size_t i = b.capacity_in_bits(); i-- > 0;) {
				want += b.test(i) ? '1' : '0';
			}
			REQUIRE_EQ(want, "000001000000001000000001");

			CHECK_EQ(binary_digits(std::format("{:b}", b)), want);
		}
	}

	// The remaining three run in a subprocess: the proxy's noexcept accessors call test()/set(),
	// which throw for an index the capped storage rejects, so the throw becomes std::terminate.
	// A catchable failure would let the child return 0 instead.

	TEST_CASE("proxy operator bool throws through noexcept") {
		// r3892406755
		auto const res = DICE_SANDBOX {
			capped8 b{};
			b.set(9);  // logical size 10, capacity 16
			auto const it = b.begin() + 12;

			try {
				if (static_cast<bool>(*it)) {
					return 1;
				}
			} catch (std::out_of_range const &) {
				return 0;
			}
			return 0;
		};

		CHECK_EQ(res, SubProcessResult::ExitSuccess);
	}

	TEST_CASE("proxy assignment throws through noexcept") {
		// r3892404711
		auto const res = DICE_SANDBOX {
			capped8 b{};
			b.set(9);
			auto const it = b.begin() + 12;

			try {
				*it = true;
			} catch (std::out_of_range const &) {
				return 0;
			}
			return 0;
		};

		CHECK_EQ(res, SubProcessResult::ExitSuccess);
	}

	TEST_CASE("format {:b} does not terminate the process") {
		// r3892440968: the formatter used to read past the logical end through the bit proxy (see
		// "formatter" above), and on a capped bitset those indices make test() throw out of a
		// noexcept proxy - i.e. std::terminate. It reads segments directly now.
		SUBCASE("capped") {
			auto const res = DICE_SANDBOX {
				capped8 b{};
				b.set(9);

				try {
					(void) std::format("{:b}", b);
				} catch (std::out_of_range const &) {
					return 0;
				}
				return 0;
			};

			CHECK_EQ(res, SubProcessResult::ExitSuccess);
		}

		SUBCASE("fixed") {
			auto const res = DICE_SANDBOX {
				fixed8 b{0x00, 0x00};

				try {
					(void) std::format("{:b}", b);
				} catch (std::out_of_range const &) {
					return 0;
				}
				return 0;
			};

			CHECK_EQ(res, SubProcessResult::ExitSuccess);
		}
	}
}

// Differential tests: drive a bitset and a reference implementation through the same random
// operation stream and compare every observable after each step. std::bitset<N> is the reference
// for fully static bitsets, boost::dynamic_bitset for the growable ones (it is the only reference
// with the same "logical size grows on demand" model).
//
// Seeds are fixed, so a failure is reproducible. This is how the extra findings above were
// located; everything asserted here passes today, so it doubles as a regression net for the
// arithmetic that hand-written expectations cannot cover exhaustively.
TEST_SUITE("bitset differential") {
	using namespace dice::template_library;

	using dyn8 = bitset<dynamic_extent, dynamic_extent, uint8_t>;

	// counts a run of equal bits from one end of the reference, capped at its size
	template<typename Ref>
	size_t run_length(Ref const &ref, bool const want, bool const from_msb) {
		for (size_t i = 0; i < ref.size(); ++i) {
			if (ref.test(from_msb ? ref.size() - 1 - i : i) != want) {
				return i;
			}
		}
		return ref.size();
	}

	template<typename Sut, typename Ref>
	void compare(Sut const &sut, Ref const &ref, size_t const logical) {
		REQUIRE_EQ(sut.size_in_bits(), logical);
		for (size_t i = 0; i < logical; ++i) {
			REQUIRE_EQ(sut.test(i), ref.test(i));
		}
		REQUIRE_EQ(sut.count(), ref.count());
		REQUIRE_EQ(sut.all_set(), ref.all());
		REQUIRE_EQ(sut.any_set(), ref.any());
		REQUIRE_EQ(sut.none_set(), ref.none());

		REQUIRE_EQ(sut.countr_zero(), run_length(ref, false, false));
		REQUIRE_EQ(sut.countl_zero(), run_length(ref, false, true));
		REQUIRE_EQ(sut.countr_one(), run_length(ref, true, false));
		REQUIRE_EQ(sut.countl_one(), run_length(ref, true, true));

		std::vector<size_t> want;
		for (size_t i = 0; i < logical; ++i) {
			if (ref.test(i)) {
				want.push_back(i);
			}
		}
		std::vector<size_t> got;
		for (auto const pos : sut.positions()) {
			got.push_back(pos);
		}
		REQUIRE(got == want);

		// forward and reverse iteration must both cover exactly the logical bits
		std::vector<bool> forward;
		for (bool const b : sut) {
			forward.push_back(b);
		}
		REQUIRE_EQ(forward.size(), logical);
		for (size_t i = 0; i < logical; ++i) {
			REQUIRE_EQ(forward[i], ref.test(i));
		}

		size_t reverse_seen = 0;
		// named rather than auto, so every instantiation has to expose the alias under this name
		for (typename Sut::const_reverse_iterator it = sut.rbegin(); it != sut.rend(); ++it, ++reverse_seen) {
			REQUIRE_EQ(static_cast<bool>(*it), ref.test(logical - 1 - reverse_seen));
		}
		REQUIRE_EQ(reverse_seen, logical);

		REQUIRE(sut.capacity_in_bits() >= sut.size_in_bits());
		REQUIRE(sut.count() <= sut.size_in_bits());
	}

	// flips a random subset of an operand pair in lockstep, so the two stay equal-sized
	template<typename Sut, typename Ref, typename Rng>
	std::pair<Sut, Ref> mixed_operand(Sut const &sut, Ref const &ref, Rng &rng, size_t const logical) {
		auto sut_other = sut;
		auto ref_other = ref;
		for (size_t i = 0; i < logical; ++i) {
			if (rng() % 2 == 0) {
				sut_other.flip(i);
				ref_other.flip(i);
			}
		}
		return {sut_other, ref_other};
	}

	TEST_CASE_TEMPLATE("fully static bitsets match std::bitset", Seg, uint8_t, uint16_t, uint64_t) {
		static constexpr size_t bits = 8 * sizeof(Seg) + 2;  // two segments plus a 2-bit remainder
		using sut_type = bitset<bits, bits, Seg>;

		sut_type sut{};
		std::bitset<bits> ref{};
		std::mt19937 rng{1000u + static_cast<unsigned>(sizeof(Seg))};

		compare(sut, ref, bits);

		for (int step = 0; step < 600; ++step) {
			auto const ix = rng() % bits;
			auto const shift = rng() % (bits + 3);

			switch (rng() % 11) {
				case 0: sut.set(ix), ref.set(ix); break;
				case 1: sut.reset(ix), ref.reset(ix); break;
				case 2: sut.flip(ix), ref.flip(ix); break;
				case 3: sut.set_all(), ref.set(); break;
				case 4: sut.reset_all(), ref.reset(); break;
				case 5: sut <<= shift, ref <<= shift; break;
				case 6: sut >>= shift, ref >>= shift; break;
				case 7: sut = ~sut, ref = ~ref; break;
				default: {
					auto const [sut_other, ref_other] = mixed_operand(sut, ref, rng, bits);
					if (rng() % 3 == 0) {
						sut &= sut_other, ref &= ref_other;
					} else if (rng() % 2 == 0) {
						sut |= sut_other, ref |= ref_other;
					} else {
						sut ^= sut_other, ref ^= ref_other;
					}
					break;
				}
			}

			compare(sut, ref, bits);
		}
	}

	TEST_CASE_TEMPLATE("growable bitsets match boost::dynamic_bitset", Seg, uint8_t, uint16_t, uint64_t) {
		using sut_type = bitset<dynamic_extent, dynamic_extent, Seg>;
		static constexpr size_t seg_bits = 8 * sizeof(Seg);

		sut_type sut{};
		boost::dynamic_bitset<Seg> ref{};
		std::mt19937 rng{2000u + static_cast<unsigned>(sizeof(Seg))};

		compare(sut, ref, 0);

		for (int step = 0; step < 400; ++step) {
			// mix small steps with jumps far past the end, to exercise multi-segment growth
			auto const span = (rng() % 4 == 0) ? ref.size() + 200 : ref.size() + 8;
			auto const ix = rng() % std::max<size_t>(span, 8);
			auto const shift = (rng() % 2 == 0) ? seg_bits * (rng() % 4) : rng() % (3 * ref.size() + 16);

			auto const grow_to = [&ref](size_t const n) {
				if (n >= ref.size()) {
					ref.resize(n + 1, false);
				}
			};

			switch (rng() % 12) {
				case 0: sut.set(ix), grow_to(ix), ref.set(ix); break;
				case 1: sut.set(ix, false), grow_to(ix), ref.reset(ix); break;
				case 2:
					sut.reset(ix);
					if (ix < ref.size()) {
						ref.reset(ix);
					}
					break;
				case 3:
					// flip() grows the logical size exactly like set() does
					sut.flip(ix), grow_to(ix), ref.flip(ix);
					break;
				case 4: sut.set_all(), ref.set(); break;
				case 5: sut.reset_all(), ref.reset(); break;
				case 6: {
					// <<= grows the logical size by the shift, capped at the capacity (the storage
					// never grows), so the reference has to be resized before it shifts
					ref.resize(std::min(ref.size() + shift, sut.capacity_in_bits()), false);
					sut <<= shift, ref <<= shift;
					break;
				}
				case 7:
					// >>= shrinks the logical size by the shift
					sut >>= shift, ref >>= shift;
					ref.resize(shift > ref.size() ? 0 : ref.size() - shift, false);
					break;
				case 8: sut = ~sut, ref = ~ref; break;
				case 9:
				case 10: {
					auto const [sut_other, ref_other] = mixed_operand(sut, ref, rng, ref.size());
					if (rng() % 3 == 0) {
						sut &= sut_other, ref &= ref_other;
					} else if (rng() % 2 == 0) {
						sut |= sut_other, ref |= ref_other;
					} else {
						sut ^= sut_other, ref ^= ref_other;
					}
					break;
				}
				default: {
					// first index that is still low, or one past the end when the set is full
					size_t want = ref.size();
					for (size_t i = 0; i < ref.size(); ++i) {
						if (!ref.test(i)) {
							want = i;
							break;
						}
					}
					if (want == ref.size()) {
						ref.resize(want + 1, false);
					}
					ref.set(want);
					REQUIRE_EQ(sut.set_first_free(), want);
					break;
				}
			}

			compare(sut, ref, ref.size());
		}
	}

	TEST_CASE_TEMPLATE("capped bitsets match boost::dynamic_bitset", Seg, uint8_t, uint64_t) {
		static constexpr size_t max_bits = 8 * sizeof(Seg) + 2;
		using sut_type = bitset<dynamic_extent, max_bits, Seg>;

		sut_type sut{};
		boost::dynamic_bitset<Seg> ref{};
		std::mt19937 rng{3000u + static_cast<unsigned>(sizeof(Seg))};

		compare(sut, ref, 0);

		for (int step = 0; step < 400; ++step) {
			auto const ix = rng() % (max_bits + 4);  // deliberately reaches past the cap
			auto const shift = rng() % (max_bits + 3);

			auto const grow_to = [&ref](size_t const n) {
				if (n >= ref.size()) {
					ref.resize(n + 1, false);
				}
			};

			switch (rng() % 10) {
				case 0:
				case 1:
					// past the cap set() must throw and leave the bitset untouched
					if (ix >= max_bits) {
						CHECK_THROWS_AS(sut.set(ix), std::out_of_range);
					} else {
						sut.set(ix), grow_to(ix), ref.set(ix);
					}
					break;
				case 2:
					// past the cap every mutator throws, exactly like set()
					if (ix >= max_bits) {
						CHECK_THROWS_AS(sut.reset(ix), std::out_of_range);
					} else {
						sut.reset(ix);
						if (ix < ref.size()) {
							ref.reset(ix);
						}
					}
					break;
				case 3:
					if (ix >= max_bits) {
						CHECK_THROWS_AS(sut.flip(ix), std::out_of_range);
					} else {
						sut.flip(ix), grow_to(ix), ref.flip(ix);
					}
					break;
				case 4: sut.set_all(), ref.set(); break;
				case 5: sut.reset_all(), ref.reset(); break;
				case 6: {
					// <<= grows the logical size by the shift, capped at what the bitset can hold -
					// the storage never grows, and it never reaches past max_bits
					ref.resize(std::min({ref.size() + shift, sut.capacity_in_bits(), max_bits}), false);
					sut <<= shift, ref <<= shift;
					break;
				}
				case 7:
					// >>= shrinks the logical size by the shift
					sut >>= shift, ref >>= shift;
					ref.resize(shift > ref.size() ? 0 : ref.size() - shift, false);
					break;
				case 8: sut = ~sut, ref = ~ref; break;
				default: {
					auto const [sut_other, ref_other] = mixed_operand(sut, ref, rng, ref.size());
					sut ^= sut_other, ref ^= ref_other;
					break;
				}
			}

			compare(sut, ref, ref.size());
			REQUIRE(sut.size_in_bits() <= max_bits);
		}
	}

	TEST_CASE("range algorithms work through the bit proxy") {
		SUBCASE("fill / count / equal / find") {
			dyn8 b{};
			b.set(9);  // 10 logical bits
			std::ranges::fill(b, true);
			CHECK_EQ(b.count(), 10);
			CHECK(b.all_set());

			b.reset_all();
			b.set(1);
			b.set(9);
			CHECK_EQ(std::ranges::count(b, true), 2);
			CHECK_EQ((*std::ranges::find(b, true)).ix(), 1);

			std::vector<bool> const want{false, true, false, false, false, false, false, false, false, true};
			CHECK(std::ranges::equal(b, want));
		}

		SUBCASE("copy into the bitset") {
			dyn8 b{};
			b.set(9);
			b.reset(9);

			std::vector<bool> const src{true, false, true, true, false, false, false, false, true, true};
			std::ranges::copy(src, b.begin());
			CHECK(std::ranges::equal(b, src));
			CHECK_EQ(b.count(), 5);
		}

		SUBCASE("reverse and iter_swap permute bits in place") {
			dyn8 b{};
			b.set(9);
			b.set(1);
			b.set(0);

			std::ranges::reverse(b);
			CHECK_EQ(b.count(), 3);
			CHECK(b.test(0));  // was bit 9
			CHECK(b.test(8));  // was bit 1
			CHECK(b.test(9));  // was bit 0

			std::ranges::iter_swap(b.begin(), b.begin() + 1);
			CHECK_FALSE(b.test(0));
			CHECK(b.test(1));
		}

		SUBCASE("proxy assignment copies the value, not the binding") {
			dyn8 b{};
			b.set(9);
			b.set(0);

			*(b.begin() + 1) = *b.begin();
			CHECK(b.test(1));
			CHECK_EQ(b.count(), 3);
		}
	}
}

