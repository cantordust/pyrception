import pytest
import enum

from pyrception.utils.enums import (
    StrEnum,
    LogTint,
    InputType,
    RFArrangement,
    KernelFilter,
    KernelShape,
)


class TestStrEnum:
    """Tests for the custom StrEnum base class with case-insensitive lookup."""

    def test_case_insensitive_lookup(self):
        assert InputType("image") == InputType.Image
        assert InputType("IMAGE") == InputType.Image
        assert InputType("Image") == InputType.Image

    def test_missing_returns_none_for_invalid(self):
        with pytest.raises(ValueError):
            InputType("nonexistent")

    def test_accepts_own_member(self):
        assert InputType(InputType.Image) == InputType.Image


class TestLogTint:
    def test_members_exist(self):
        expected = {"trace", "debug", "info", "success", "warning", "error", "critical"}
        assert set(m.name for m in LogTint) == expected

    def test_values_are_strings(self):
        for member in LogTint:
            assert isinstance(member.value, str)


class TestInputType:
    def test_members(self):
        assert set(m.name for m in InputType) == {"Image", "Video", "Events"}

    def test_auto_values(self):
        for member in InputType:
            assert isinstance(member.value, str)


class TestRFArrangement:
    def test_members(self):
        assert set(m.name for m in RFArrangement) == {"LogPolar", "Cartesian"}


class TestKernelFilter:
    def test_members(self):
        assert set(m.name for m in KernelFilter) == {"Uniform", "Gaussian", "Gabor"}


class TestKernelShape:
    def test_members(self):
        assert set(m.name for m in KernelShape) == {"Elliptic", "Rectangular"}
