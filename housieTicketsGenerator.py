import os
import random
from PIL import Image, ImageDraw, ImageFont

def get_valid_ticket_grid():
    """Generates a valid 3x9 grid layout."""
    while True:
        grid = [[0] * 9 for _ in range(3)]

        for col in range(9):
            grid[random.randint(0, 2)][col] = 1

        candidates = [
            (r, c)
            for r in range(3)
            for c in range(9)
            if grid[r][c] == 0
        ]

        random.shuffle(candidates)

        def backtrack(idx, count_added):
            if count_added == 6:
                return (
                    all(sum(grid[r]) == 5 for r in range(3)) and
                    all(
                        1 <= sum(grid[r][c] for r in range(3)) <= 3
                        for c in range(9)
                    )
                )

            if idx >= len(candidates):
                return False

            r, c = candidates[idx]

            if sum(grid[r]) < 5 and sum(grid[r2][c] for r2 in range(3)) < 3:
                grid[r][c] = 1

                if backtrack(idx + 1, count_added + 1):
                    return True

                grid[r][c] = 0

            return backtrack(idx + 1, count_added)

        if backtrack(0, 0):
            return grid


def generate_single_housie_ticket():
    """Generates a single Housie ticket."""
    grid = get_valid_ticket_grid()

    ticket = [[None] * 9 for _ in range(3)]

    col_ranges = [
        (1, 9), (10, 19), (20, 29), (30, 39), (40, 49),
        (50, 59), (60, 69), (70, 79), (80, 90)
    ]

    for c in range(9):
        active_rows = [r for r in range(3) if grid[r][c] == 1]

        low, high = col_ranges[c]

        nums = sorted(
            random.sample(range(low, high + 1), len(active_rows))
        )

        for i, r in enumerate(active_rows):
            ticket[r][c] = nums[i]

    return ticket


def validate_housie_ticket(ticket):
    """Checks whether the ticket follows Housie rules."""

    assert len(ticket) == 3
    assert all(len(row) == 9 for row in ticket)

    for r in range(3):
        assert sum(1 for val in ticket[r] if val is not None) == 5

    col_ranges = [
        (1, 9), (10, 19), (20, 29), (30, 39), (40, 49),
        (50, 59), (60, 69), (70, 79), (80, 90)
    ]

    all_numbers = []

    for c in range(9):
        col_vals = [
            ticket[r][c]
            for r in range(3)
            if ticket[r][c] is not None
        ]

        assert 1 <= len(col_vals) <= 3

        low, high = col_ranges[c]

        for val in col_vals:
            assert low <= val <= high
            all_numbers.append(val)

        assert col_vals == sorted(col_vals)

    assert len(all_numbers) == 15
    assert len(all_numbers) == len(set(all_numbers))


def render_tickets_image(
    tickets,
    output_filepath=os.path.join("Output_images", "housie_tickets.png")
):
    """Renders all tickets into one PNG image."""

    output_dir = os.path.dirname(output_filepath)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cell_width = 80
    cell_height = 70

    grid_width = cell_width * 9
    grid_height = cell_height * 3

    ticket_header_height = 45
    ticket_padding_v = 25

    ticket_total_height = (
        ticket_header_height +
        grid_height +
        ticket_padding_v
    )

    margin_x = 50
    margin_top = 100
    margin_bottom = 50

    image_width = grid_width + (margin_x * 2)
    image_height = (
        margin_top +
        len(tickets) * ticket_total_height +
        margin_bottom
    )

    # Colors
    bg_color = (255, 248, 240)
    card_bg = (255, 255, 255)
    empty_cell_bg = (248, 235, 225)

    header_colors = [
        (166, 54, 54),
        (180, 76, 60),
        (145, 45, 45),
        (190, 100, 55),
        (155, 65, 50)
    ]

    border_color = (120, 70, 55)
    grid_line_color = (220, 195, 180)

    text_color = (70, 35, 25)
    title_color = (110, 45, 35)
    subtitle_color = (130, 100, 90)

    image = Image.new("RGB", (image_width, image_height), bg_color)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        subtitle_font = ImageFont.truetype("arial.ttf", 16)
        header_font = ImageFont.truetype("arialbd.ttf", 20)
        number_font = ImageFont.truetype("arialbd.ttf", 30)
    except IOError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    # Title
    title_text = "HOUSIE / TAMBOLA TICKETS"

    title_bbox = draw.textbbox(
        (0, 0),
        title_text,
        font=title_font
    )

    title_width = title_bbox[2] - title_bbox[0]

    draw.text(
        ((image_width - title_width) // 2, 25),
        title_text,
        fill=title_color,
        font=title_font
    )

    # Subtitle
    sub_text = "Standard 3x9 Tickets • 15 Numbers Per Ticket • 5 Numbers Per Row"

    sub_bbox = draw.textbbox(
        (0, 0),
        sub_text,
        font=subtitle_font
    )

    sub_width = sub_bbox[2] - sub_bbox[0]

    draw.text(
        ((image_width - sub_width) // 2, 65),
        sub_text,
        fill=subtitle_color,
        font=subtitle_font
    )

    # Draw tickets
    for idx, ticket in enumerate(tickets):

        top_y = margin_top + idx * ticket_total_height
        left_x = margin_x

        header_bg = header_colors[idx % len(header_colors)]

        # Header
        draw.rectangle(
            [
                left_x,
                top_y,
                left_x + grid_width,
                top_y + ticket_header_height
            ],
            fill=header_bg
        )

        ticket_title = f"TICKET #{idx + 1}"

        h_bbox = draw.textbbox(
            (0, 0),
            ticket_title,
            font=header_font
        )

        h_height = h_bbox[3] - h_bbox[1]

        draw.text(
            (
                left_x + 15,
                top_y + (ticket_header_height - h_height) // 2
            ),
            ticket_title,
            fill=(255, 255, 255),
            font=header_font
        )

        # Grid
        grid_top_y = top_y + ticket_header_height

        for r in range(3):
            for c in range(9):

                cell_x1 = left_x + c * cell_width
                cell_y1 = grid_top_y + r * cell_height

                cell_x2 = cell_x1 + cell_width
                cell_y2 = cell_y1 + cell_height

                val = ticket[r][c]

                if val is None:
                    draw.rectangle(
                        [cell_x1, cell_y1, cell_x2, cell_y2],
                        fill=empty_cell_bg,
                        outline=grid_line_color
                    )

                else:
                    draw.rectangle(
                        [cell_x1, cell_y1, cell_x2, cell_y2],
                        fill=card_bg,
                        outline=grid_line_color
                    )

                    text = str(val)

                    bbox = draw.textbbox(
                        (0, 0),
                        text,
                        font=number_font
                    )

                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]

                    draw.text(
                        (
                            cell_x1 + (cell_width - w) // 2,
                            cell_y1 + (cell_height - h) // 2 - 3
                        ),
                        text,
                        fill=text_color,
                        font=number_font
                    )

        # Outer border
        draw.rectangle(
            [
                left_x,
                top_y,
                left_x + grid_width,
                grid_top_y + grid_height
            ],
            outline=border_color,
            width=2
        )

    image.save(output_filepath)

    print(
        f"Successfully rendered {len(tickets)} Housie tickets to: "
        f"'{output_filepath}'"
    )


def main():
    tickets = []

    for i in range(5):
        ticket = generate_single_housie_ticket()
        validate_housie_ticket(ticket)
        tickets.append(ticket)

    output_path = os.path.join(
        "Output_images",
        "housie_tickets.png"
    )

    render_tickets_image(tickets, output_path)


if __name__ == "__main__":
    main()