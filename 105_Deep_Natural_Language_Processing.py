from chunker import PennTreebackChunker
from extractor import SOPExtractor

chunker = PennTreebackChunker()
extractor = SOPExtractor(chunker)

def extract(sentence):
    sentence = sentence if sentence[-1] == '.' else sentence+'.'
    global extractor
    sop_triplet = extractor.extract(sentence)
    return sop_triplet


sentences = [
  'The quick brown fox jumps over the lazy dog.',
  'A rare black squirrel has become a regular visitor to a suburban garden',
  'The driver did not change the flat tire',
  "The driver crashed the bike white bumper"
]

for sentence in sentences:
    sop_triplet = extract(sentence)
    print (sop_triplet.subject + ':' + sop_triplet.predicate + ':' + sop_triplet.object)