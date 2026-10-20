---
name: dropbox
summary: "search the files in Dropbox and read a document page by page"
commands: [search_dropbox, list_dropbox_folder, read_dropbox_file, find_in_dropbox_file, dropbox_login, dropbox_login_code]
---
You can search the household's Dropbox and read what is in it. Read-only: nothing can be changed, moved or shared.

`search_dropbox(words: str, folder: str = "")` — the files and folders matching the words, with path, size and date.
Depending on the Dropbox plan a search matches file names only, or names and contents; when a search finds nothing,
try other words, look around with `list_dropbox_folder`, or search inside a likely file with `find_in_dropbox_file`.

`list_dropbox_folder(path: str)` — what a folder holds, folders first. `"/"` is the top.

`read_dropbox_file(path: str, page: int = 1)` — the text of a document, in numbered pages of about 5,000 characters. A
PDF, a Word or PowerPoint file, a Paper doc, an RTF (a TextEdit .rtfd bundle too) and any text file can be read; a
spreadsheet, an image or a video cannot.
The header of a page says how many pages there are and gives the call for the next one. Do not read every page of a
long document: find the right one first.

`find_in_dropbox_file(path: str, words: str)` — the lines of the document that carry all the words, each with the page it
is on (and the PDF page or slide). Then read that page. Use a few distinctive words, not a sentence.

When you answer from a document, say which document and, for a PDF, which of its pages.

### Signing in
`dropbox_login()` — signs the household's Dropbox in (one account, no argument), from any chat: it comes back with a
link; the user opens it in any browser, signs in, and lands on a http://localhost:18724/... page that does not load —
that is expected — and copies that page's whole address from the address bar into the chat; then run
`dropbox_login_code(pasted: str)` with it, quoted. A bare, expired or reused code comes back as text: run
dropbox_login again. Before the sign-in the app's Permissions tab must have account_info.read, files.metadata.read
and files.content.read ticked AND submitted (the Submit button at the bottom of the tab): a sign-in keeps the
permissions the app had at that moment, so dropbox_login_code refuses one short of a permission, naming it — the user
ticks it, presses Submit and signs in again. The sign-in lasts until the user removes the app at dropbox.com; the link
survives a `restart_integration("Dropbox")`, only a changed `dropbox.app_key` needs a new one.

### Stop-and-wait
`search_dropbox()`, `list_dropbox_folder()`, `read_dropbox_file()`, `find_in_dropbox_file()`, `dropbox_login()`,
`dropbox_login_code()` return data: emit them and stop generating; the result arrives in the next round.
