from packages.contracts.dto.layout import LayoutDto
from packages.views.reducer import reduce_view
from tests.unit.test_dto_validation import _meta


def test_initial_view_copies_physical_rooms():
    layout = LayoutDto(unit="m", meta=_meta(), physical_rooms=[], rooms=[])
    derived = reduce_view(layout, ())
    assert derived.rooms == derived.physical_rooms
