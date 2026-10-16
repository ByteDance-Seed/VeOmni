"""``ConversationItem`` routing: items are selected by data tags, never by module."""

import torch

from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem, iter_desired_items


def test_a_dummy_is_selected_by_the_same_filter_as_the_items_it_stands_in_for():
    """An encoder reads real and placeholder rows with one filter and tells them
    apart by ``is_dummy``; a placeholder tagged for another encoder is not taken."""
    und = ConversationItem(type="image", value=torch.zeros(1), role="user", meta={_IMG_TAG_KEY: "und"})
    und_dummy = ConversationItem(
        type="image", value=torch.zeros(1), role="user", is_dummy=True, meta={_IMG_TAG_KEY: "und"}
    )
    gen_dummy = ConversationItem(
        type="image", value=torch.zeros(1), role="assistant", is_dummy=True, meta={_IMG_TAG_KEY: "gen"}
    )
    conversation_list = [[und], [und_dummy, gen_dummy]]

    picked = list(iter_desired_items(conversation_list, types=["image"], roles=["user"], meta={_IMG_TAG_KEY: ["und"]}))

    assert picked == [und, und_dummy]
    assert [item.is_dummy for item in picked] == [False, True]
